"""
Step 2: Generate image descriptions from 3 independent proprietary VLMs.

Takes the classification CSV (math images only) and sends each image to
OpenAI, Gemini, and Claude independently, using the same structured checklist
prompt with a category-specific hint.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from . import config
from .api_clients import (
    get_openai_client,
    get_gemini_model,
    get_claude_client,
    get_qwen_client,
    get_llama_client,
)
from .data_loader import load_mathvision, get_image_pil, get_image_base64
from .utils import encode_image_base64, call_with_retry, load_checkpoint, save_checkpoint


logger = logging.getLogger(__name__)

def _build_description_prompt(category: str = "unknown") -> str:
    """Build the full description prompt with category-specific hint."""
    base = config.DETAILED_DESCRIPTION_PROMPT
    hint = config.CATEGORY_HINTS.get(category, "")
    if hint:
        return f"{base}\n\n{hint}"
    return base


def get_openai_description(client, image_b64: str, question_text: str, category: str = "unknown") -> str:
    """Get description from OpenAI."""
    prompt = _build_description_prompt(category)

    def _call():
        response = client.chat.completions.create(
            model=config.OPENAI_DESCRIPTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt + "\n\nDO NOT solve the problem or answer the question. ONLY describe what you see in the image visually.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Image category: {category}\nQuestion context (do NOT answer): {question_text}\n\nDescribe every visual element in this mathematical diagram.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_completion_tokens=config.DESCRIPTION_MAX_TOKENS,
            timeout=60,
        )
        return response.choices[0].message.content

    start = time.time()
    result = call_with_retry(_call)
    elapsed_ms = (time.time() - start) * 1000
    response_len = len(result) if isinstance(result, str) else 0
    logger.info(
        "[TIMING] openai.describe response_ms=%.1f response_len=%d",
        elapsed_ms,
        response_len,
    )
    if isinstance(result, str) and ("[API ERROR" in result or "[QUOTA" in result):
        return f"[OPENAI ERROR: {result}]"
    return result


def get_gemini_description(model, image_pil, question_text: str, category: str = "unknown") -> str:
    """Get description from Gemini."""
    prompt = _build_description_prompt(category)

    def _call():
        full_prompt = [
            prompt + f"\n\nDO NOT solve the problem or answer the question. ONLY describe what you see.\n\nImage category: {category}\nQuestion context (do NOT answer): {question_text}\n\nDescribe every visual element.",
            image_pil,
        ]
        response = model.generate_content(
            full_prompt,
            request_options={"timeout": 180},
        )
        return response.text

    start = time.time()
    result = call_with_retry(_call)
    elapsed_ms = (time.time() - start) * 1000
    response_len = len(result) if isinstance(result, str) else 0
    logger.info(
        "[TIMING] gemini.describe response_ms=%.1f response_len=%d",
        elapsed_ms,
        response_len,
    )
    if isinstance(result, str) and ("[API ERROR" in result or "[QUOTA" in result):
        return f"[GEMINI ERROR: {result}]"
    return result


def get_claude_description(client, image_b64: str, question_text: str, category: str = "unknown") -> str:
    """Get description from Claude."""
    prompt = _build_description_prompt(category)

    def _call():
        response = client.messages.create(
            model=config.CLAUDE_DESCRIPTION_MODEL,
            max_tokens=config.DESCRIPTION_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{prompt}\n\n"
                                f"DO NOT solve the problem or answer the question. ONLY describe what you see visually.\n\n"
                                f"Image category: {category}\n"
                                f"Question context (do NOT answer): {question_text}\n\n"
                                f"Describe every visual element in this mathematical diagram."
                            ),
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                    ],
                },
            ],
        )
        return response.content[0].text

    start = time.time()
    result = call_with_retry(_call)
    elapsed_ms = (time.time() - start) * 1000
    response_len = len(result) if isinstance(result, str) else 0
    logger.info(
        "[TIMING] claude.describe response_ms=%.1f response_len=%d",
        elapsed_ms,
        response_len,
    )
    if isinstance(result, str) and ("[API ERROR" in result or "[QUOTA" in result):
        return f"[CLAUDE ERROR: {result}]"
    return result


def get_qwen_description(client, image_b64: str, question_text: str, category: str = "unknown") -> str:
    """Get description from Qwen vision model via OpenRouter."""
    prompt = _build_description_prompt(category)

    def _call():
        response = client.chat.completions.create(
            model=config.QWEN_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt + "\n\nDO NOT solve the problem or answer the question. ONLY describe what you see in the image visually.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Image category: {category}\nQuestion context (do NOT answer): {question_text}\n\nDescribe every visual element in this mathematical diagram.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_tokens=config.DESCRIPTION_MAX_TOKENS,
            timeout=60,
        )
        return response.choices[0].message.content

    start = time.time()
    result = call_with_retry(_call)
    elapsed_ms = (time.time() - start) * 1000
    response_len = len(result) if isinstance(result, str) else 0
    logger.info(
        "[TIMING] qwen.describe response_ms=%.1f response_len=%d",
        elapsed_ms,
        response_len,
    )
    if isinstance(result, str) and ("[API ERROR" in result or "[QUOTA" in result):
        return f"[QWEN ERROR: {result}]"
    return result


def get_llama_description(client, image_b64: str, question_text: str, category: str = "unknown") -> str:
    """Get description from Llama vision model via Groq."""
    prompt = _build_description_prompt(category)

    def _call():
        response = client.chat.completions.create(
            model=config.LLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt + "\n\nDO NOT solve the problem or answer the question. ONLY describe what you see in the image visually.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Image category: {category}\nQuestion context (do NOT answer): {question_text}\n\nDescribe every visual element in this mathematical diagram.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_tokens=config.DESCRIPTION_MAX_TOKENS,
            timeout=60,
        )
        return response.choices[0].message.content

    start = time.time()
    result = call_with_retry(_call)
    elapsed_ms = (time.time() - start) * 1000
    response_len = len(result) if isinstance(result, str) else 0
    logger.info(
        "[TIMING] llama.describe response_ms=%.1f response_len=%d",
        elapsed_ms,
        response_len,
    )
    if isinstance(result, str) and ("[API ERROR" in result or "[QUOTA" in result):
        return f"[LLAMA ERROR: {result}]"
    return result


def _is_valid_description(text) -> bool:
    """Check if a description is usable (not an error message)."""
    if not text or not isinstance(text, str):
        return False
    if pd.isna(text):
        return False
    if len(text.strip()) < 50:
        return False
    error_markers = [
        "[ERROR",
        "[OPENAI ERROR",
        "[GEMINI ERROR",
        "[CLAUDE ERROR",
        "[QWEN ERROR",
        "[LLAMA ERROR",
        "[API ERROR",
        "[QUOTA",
        "nan",
    ]
    head = text.strip()[:40]
    return not any(marker in head for marker in error_markers)


def run_description(
    input_csv: str | None = None,
    resume: bool = True,
    delay: float | None = None,
    workers: int = 5,
):
    """
    Generate descriptions for all math-classified images using multiple providers.

    Args:
        input_csv: Path to classification CSV. Defaults to config.CLASSIFICATION_CSV.
        resume: Resume from existing checkpoint.
        delay: Seconds between API calls.

    Returns:
        DataFrame with columns: image_id, question, category,
        description_openai, description_gemini, description_claude,
        description_qwen, description_llama, error_log
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

    # Check for category data
    has_categories = "final_category" in df.columns
    if has_categories:
        print("Category-aware prompting enabled:")
        for cat, count in df["final_category"].value_counts().items():
            print(f"  {cat}: {count}")
    else:
        print("No category data — using generic prompt for all images")

    results, processed_ids = load_checkpoint(config.DESCRIPTIONS_CSV) if resume else ([], set())
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} already described")

    # Initialize clients
    openai_client = get_openai_client()
    gemini_model = get_gemini_model()
    claude_client = get_claude_client()
    qwen_client = get_qwen_client()
    llama_client = get_llama_client()

    processed_count = 0

    providers = ["openai", "gemini", "claude", "qwen", "llama"]

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Describing ({len(providers)} providers)",
    ):
        image_id = row["image_id"]
        if image_id in processed_ids:
            continue

        question_text = row["question"]
        category = str(row.get("final_category", row.get("majority_category", "unknown")))
        if category == "nan" or pd.isna(category):
            category = "unknown"

        entry = {
            "image_id": image_id,
            "question": question_text,
            "category": category,
            "description_openai": "",
            "description_gemini": "",
            "description_claude": "",
            "description_qwen": "",
            "description_llama": "",
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

        errors = []

        time.sleep(delay)

        provider_calls = {
            "openai": (get_openai_description, openai_client, img_b64, question_text, category),
            "gemini": (get_gemini_description, gemini_model, img_pil, question_text, category),
            "claude": (get_claude_description, claude_client, img_b64, question_text, category),
            "qwen":   (get_qwen_description,   qwen_client,   img_b64, question_text, category),
            "llama":  (get_llama_description,  llama_client,  img_b64, question_text, category),
        }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                name: pool.submit(fn, *args)
                for name, (fn, *args) in provider_calls.items()
            }
            for name, future in futures.items():
                try:
                    entry[f"description_{name}"] = future.result()
                except Exception as e:
                    entry[f"description_{name}"] = f"[ERROR: {str(e)[:200]}]"
                    errors.append(f"{name}: {str(e)[:100]}")

        entry["error_log"] = "; ".join(errors) if errors else ""

        results.append(entry)
        processed_ids.add(image_id)
        processed_count += 1

        if processed_count % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(results, config.DESCRIPTIONS_CSV)
            print(f"  Checkpoint: {processed_count} images")

    save_checkpoint(results, config.DESCRIPTIONS_CSV)
    result_df = pd.DataFrame(results)

    # Print summary
    for provider in providers:
        col = f"description_{provider}"
        valid = result_df[col].apply(lambda x: _is_valid_description(x)).sum()
        print(f"  {provider}: {valid}/{len(result_df)} successful")

    print(f"\nDescriptions complete: {len(result_df)} images")
    return result_df


if __name__ == "__main__":
    run_description()
