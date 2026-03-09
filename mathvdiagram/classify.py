"""
Step 1: Classify images as math-diagram or non-math-diagram using GPT-4o-mini.

Loads the MathVision dataset from HuggingFace, sends each image to 4o-mini
for classification, and outputs two CSVs: one for math diagrams, one for skipped.
"""

import os
import time

import pandas as pd
from tqdm import tqdm

from . import config
from .api_clients import get_openai_client
from .data_loader import load_mathvision, get_image_bytes
from .utils import encode_image_base64, call_with_retry, load_checkpoint, save_checkpoint


def classify_single_image(client, image_b64: str) -> tuple[bool, str]:
    """Classify a single image as math/non-math using GPT-4o-mini."""

    def _call():
        response = client.chat.completions.create(
            model=config.OPENAI_CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": config.CLASSIFIER_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this a mathematical diagram?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_completion_tokens=10,
            timeout=30,
        )
        return response.choices[0].message.content

    try:
        raw = call_with_retry(_call, max_retries=3)
        if isinstance(raw, str) and "[API ERROR" in raw:
            return False, raw
        normalized = raw.replace(".", "").strip().upper()
        if normalized.startswith("YES"):
            return True, raw
        if normalized.startswith("NO"):
            return False, raw
        return False, raw
    except Exception as e:
        return False, f"[CLASSIFIER ERROR: {str(e)}]"


def run_classification(
    num_samples: int | None = None,
    test_ids: list | None = None,
    resume: bool = True,
    delay: float | None = None,
):
    """
    Classify all images in the dataset.

    Args:
        num_samples: Limit to first N samples. None = all.
        test_ids: Only process these specific IDs. None = all.
        resume: Resume from existing checkpoint.
        delay: Seconds between API calls. Defaults to config value.

    Returns:
        (math_df, skipped_df) DataFrames of classified images.
    """
    delay = delay if delay is not None else config.DELAY_BETWEEN_REQUESTS

    df = load_mathvision()
    print(f"Total rows: {len(df)}")

    if test_ids:
        df = df[df["id"].isin(test_ids)].copy()
        print(f"Filtered to {len(df)} rows for test IDs: {test_ids}")
    elif num_samples:
        df = df.head(num_samples).copy()
        print(f"Limited to first {num_samples} rows")

    # Load checkpoints
    math_results, math_ids = load_checkpoint(config.CLASSIFICATION_CSV) if resume else ([], set())
    skipped_results, skipped_ids = load_checkpoint(config.SKIPPED_CSV) if resume else ([], set())
    processed_ids = math_ids | skipped_ids
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} images already processed")

    client = get_openai_client()
    processed_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        image_id = row["id"]
        if image_id in processed_ids:
            continue

        image_bytes = get_image_bytes(image_id)
        if image_bytes is None:
            print(f"  Image not found for id: {image_id}")
            math_results.append({
                "image_id": image_id,
                "question": row["question"],
                "is_math": False,
                "classifier_output": "IMAGE_NOT_FOUND",
            })
            processed_ids.add(image_id)
            processed_count += 1
            continue

        img_b64 = encode_image_base64(image_bytes)

        time.sleep(delay)
        is_math, classifier_output = classify_single_image(client, img_b64)

        entry = {
            "image_id": image_id,
            "question": row["question"],
            "is_math": is_math,
            "classifier_output": classifier_output,
        }

        if is_math:
            math_results.append(entry)
        else:
            entry["reason"] = "non_math"
            skipped_results.append(entry)

        processed_ids.add(image_id)
        processed_count += 1

        if processed_count % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(math_results, config.CLASSIFICATION_CSV)
            save_checkpoint(skipped_results, config.SKIPPED_CSV)
            print(f"  Checkpoint: {processed_count} images")

    # Final save
    save_checkpoint(math_results, config.CLASSIFICATION_CSV)
    save_checkpoint(skipped_results, config.SKIPPED_CSV)

    math_df = pd.DataFrame(math_results)
    skipped_df = pd.DataFrame(skipped_results)
    print(f"\nClassification complete: {len(math_df)} math, {len(skipped_df)} skipped")
    return math_df, skipped_df


def export_images(
    math_dir: str | None = None,
    non_math_dir: str | None = None,
):
    """
    Save classified images into separate folders (math/ and non_math/).
    Reads classification CSVs and writes images from the HF dataset to disk.
    """
    math_dir = math_dir or os.path.join(config.OUTPUT_DIR, "math")
    non_math_dir = non_math_dir or os.path.join(config.OUTPUT_DIR, "non_math")
    os.makedirs(math_dir, exist_ok=True)
    os.makedirs(non_math_dir, exist_ok=True)

    from .data_loader import load_mathvision, get_image_pil
    load_mathvision()

    saved = 0
    for csv_path, target_dir, label in [
        (config.CLASSIFICATION_CSV, math_dir, "math"),
        (config.SKIPPED_CSV, non_math_dir, "non_math"),
    ]:
        if not os.path.exists(csv_path):
            print(f"  {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {label}"):
            image_id = row["image_id"]
            img = get_image_pil(image_id)
            if img is None:
                continue
            img.save(os.path.join(target_dir, f"{image_id}.png"))
            saved += 1

    print(f"\nExported {saved} images")
    print(f"  Math:     {math_dir}")
    print(f"  Non-math: {non_math_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify math diagrams")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--export-images", action="store_true", help="Save images to math/ and non_math/ folders")
    args = parser.parse_args()

    run_classification(num_samples=args.num_samples)
    if args.export_images:
        export_images()
