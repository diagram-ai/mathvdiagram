"""
Step 2: Generate image descriptions from Gemini and OpenAI for classified math diagrams.

Takes the classification CSV (math images only) and sends each image to both
Gemini and OpenAI to get detailed visual descriptions.
"""

import os
import time

import pandas as pd
from tqdm import tqdm

from . import config
from .api_clients import get_openai_client, get_gemini_model
from .data_loader import load_mathvision, get_image_pil, get_image_base64
from .utils import encode_image_base64, call_with_retry, load_checkpoint, save_checkpoint


def get_openai_description(client, image_b64: str, prompt: str, question_text: str) -> str:
    def _call():
        response = client.chat.completions.create(
            model=config.OPENAI_DESCRIPTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt + " DO NOT solve the problem or answer the question. ONLY describe what you see in the image visually.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the visual elements in this mathematical diagram. Do not solve or answer anything.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_completion_tokens=1000,
            timeout=30,
        )
        return response.choices[0].message.content

    result = call_with_retry(_call)
    if isinstance(result, str) and result.startswith("[API ERROR"):
        return f"[OPENAI ERROR: {result}]"
    return result


def get_gemini_description(model, image_pil, prompt: str, question_text: str) -> str:
    def _call():
        full_prompt = [
            prompt,
            f"Context - Question: {question_text}",
            image_pil,
        ]
        response = model.generate_content(
            full_prompt,
            request_options={"timeout": 180},
        )
        return response.text

    result = call_with_retry(_call)
    if isinstance(result, str) and ("[API ERROR" in result or "[QUOTA" in result):
        return f"[GEMINI ERROR: {result}]"
    return result


def run_description(
    input_csv: str | None = None,
    resume: bool = True,
    delay: float | None = None,
):
    """
    Generate descriptions for all math-classified images.

    Args:
        input_csv: Path to classification CSV. Defaults to config.CLASSIFICATION_CSV.
        resume: Resume from existing checkpoint.
        delay: Seconds between API calls.

    Returns:
        DataFrame with columns: image_id, question, gemini_prompt, openai_prompt, error_log
    """
    input_csv = input_csv or config.CLASSIFICATION_CSV
    delay = delay if delay is not None else config.DELAY_BETWEEN_REQUESTS

    # Ensure HF dataset is loaded so images are available
    load_mathvision()

    print(f"Loading classified images from {input_csv}...")
    df = pd.read_csv(input_csv)
    # Only process images classified as math
    df = df[df["is_math"] == True].copy()
    print(f"Math images to describe: {len(df)}")

    # Show category distribution if taxonomy columns are present
    cat_col = "final_category" if "final_category" in df.columns else "majority_category"
    if cat_col in df.columns:
        print(f"\n  Category distribution ({cat_col}):")
        for cat, count in df[cat_col].value_counts().items():
            print(f"    {cat:<30} {count}")

    results, processed_ids = load_checkpoint(config.DESCRIPTIONS_CSV) if resume else ([], set())
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} already described")

    openai_client = get_openai_client()
    gemini_model = get_gemini_model()
    prompt = config.DETAILED_DESCRIPTION_PROMPT
    processed_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Describing"):
        image_id = row["image_id"]
        if image_id in processed_ids:
            continue

        question_text = row["question"]

        entry = {
            "image_id": image_id,
            "question": question_text,
            "gemini_prompt": "",
            "openai_prompt": "",
            "error_log": "",
        }

        img_pil = get_image_pil(image_id)
        if img_pil is None:
            entry["error_log"] = f"Image not found for id: {image_id}"
            results.append(entry)
            processed_ids.add(image_id)
            processed_count += 1
            continue

        img_b64, _ = get_image_base64(image_id)

        try:
            time.sleep(delay)
            entry["gemini_prompt"] = get_gemini_description(
                gemini_model, img_pil, prompt, question_text
            )

            time.sleep(delay)
            entry["openai_prompt"] = get_openai_description(
                openai_client, img_b64, prompt, question_text
            )
        except Exception as e:
            entry["error_log"] = str(e)

        results.append(entry)
        processed_ids.add(image_id)
        processed_count += 1

        if processed_count % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(results, config.DESCRIPTIONS_CSV)
            print(f"  Checkpoint: {processed_count} images")

    save_checkpoint(results, config.DESCRIPTIONS_CSV)
    result_df = pd.DataFrame(results)
    print(f"\nDescriptions complete: {len(result_df)} images")
    return result_df


if __name__ == "__main__":
    run_description()
