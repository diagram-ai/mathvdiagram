"""
Data loader for the MathVision dataset from HuggingFace.

Loads the dataset once and provides access to images and metadata.
Images are cached locally after first download by the `datasets` library.
"""

import io
import os
import base64

import pandas as pd
from datasets import load_dataset
from PIL import Image

from . import config

_dataset_cache = {}


def load_mathvision(split: str | None = None) -> pd.DataFrame:
    """
    Load the MathVision dataset from HuggingFace and return as a DataFrame.

    The HuggingFace dataset has columns: id, question, answer, image, level, subject, etc.
    The `image` column contains PIL Image objects.

    We store images separately in a cache and return a DataFrame with metadata.
    """
    split = split or config.HF_SPLIT
    cache_key = f"{config.HF_DATASET_NAME}_{split}"

    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]["df"]

    print(f"Loading dataset {config.HF_DATASET_NAME} (split={split}) from HuggingFace...")
    ds = load_dataset(config.HF_DATASET_NAME, split=split)
    print(f"Loaded {len(ds)} rows")

    # Store images in a dict keyed by id, build a DataFrame without the PIL objects
    images = {}
    rows = []
    for item in ds:
        img_id = item["id"]
        images[img_id] = item["decoded_image"]
        rows.append({
            "id": img_id,
            "question": item["question"],
            "answer": item.get("answer", ""),
            "level": item.get("level", ""),
            "subject": item.get("subject", ""),
        })

    df = pd.DataFrame(rows)
    _dataset_cache[cache_key] = {"df": df, "images": images}
    return df


def get_image_pil(image_id) -> Image.Image | None:
    """Get a PIL Image for a given image ID from the cached dataset."""
    key = str(image_id)
    for cache in _dataset_cache.values():
        img = cache["images"].get(key)
        if img is not None:
            return img.convert("RGB")
    return None


def get_image_base64(image_id) -> tuple[str | None, str]:
    """Get base64-encoded image bytes and media type for a given image ID."""
    img = get_image_pil(image_id)
    if img is None:
        return None, f"Image not found for id: {image_id}"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


def get_image_bytes(image_id) -> bytes | None:
    """Get raw PNG bytes for a given image ID."""
    img = get_image_pil(image_id)
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_datikz(split: str | None = None) -> pd.DataFrame:
    """
    Load the DaTikZ v3 dataset from HuggingFace and return as a DataFrame.

    Schema: caption, code, image, pdf, uri, origin.
    Images are PIL objects in the `image` column.
    """
    split = split or config.DATIKZ_SPLIT
    cache_key = f"{config.DATIKZ_DATASET_NAME}_{split}"

    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]["df"]

    print(f"Loading dataset {config.DATIKZ_DATASET_NAME} (split={split}) from HuggingFace...")
    ds = load_dataset(config.DATIKZ_DATASET_NAME, split=split)
    print(f"Loaded {len(ds)} rows")

    images = {}
    rows = []
    for i, item in enumerate(ds):
        img_id = str(i)
        images[img_id] = item["image"]
        rows.append({
            "id":      img_id,
            "caption": item.get("caption", ""),
            "code":    item.get("code", ""),
            "uri":     item.get("uri", ""),
            "origin":  item.get("origin", ""),
        })

    df = pd.DataFrame(rows)
    _dataset_cache[cache_key] = {"df": df, "images": images}
    return df


def prepare_datikz_images(
    output_csv: str | None = None,
    num_samples: int | None = None,
    test_ids: list | None = None,
) -> pd.DataFrame:
    """
    Build a pass-through CSV for the DaTikZ description step.

    Uses `caption` as the question context for VLMs (no math question exists).
    Ground truth TikZ `code` is preserved for downstream evaluation.

    Args:
        output_csv: Destination path. Defaults to config.DATIKZ_ALL_IMAGES_CSV.
        num_samples: Limit to first N rows (useful for testing).
        test_ids: Only include specific image IDs.

    Returns:
        DataFrame with columns: image_id, question, is_math, final_category, tikz_code.
    """
    output_csv = output_csv or config.DATIKZ_ALL_IMAGES_CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = load_datikz()
    if test_ids:
        df = df[df["id"].astype(str).isin([str(i) for i in test_ids])]
    elif num_samples:
        df = df.head(num_samples)

    result = pd.DataFrame({
        "image_id":       df["id"].values,
        "question":       df["caption"].values,
        "is_math":        True,
        "final_category": df["origin"].fillna("unknown").values,
        "tikz_code":      df["code"].values,
    })

    result.to_csv(output_csv, index=False)
    print(f"Prepared {len(result)} DaTikZ images → {output_csv}")
    return result


def get_datikz_image_pil(image_id) -> Image.Image | None:
    """Get a PIL Image for a given DaTikZ image ID."""
    key = str(image_id)
    cache_key = f"{config.DATIKZ_DATASET_NAME}_{config.DATIKZ_SPLIT}"
    cache = _dataset_cache.get(cache_key)
    if cache:
        img = cache["images"].get(key)
        if img is not None:
            return img.convert("RGB")
    return None


def get_datikz_image_base64(image_id) -> tuple[str | None, str]:
    """Get base64-encoded image and media type for a given DaTikZ image ID."""
    img = get_datikz_image_pil(image_id)
    if img is None:
        return None, f"DaTikZ image not found for id: {image_id}"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


def prepare_all_images(
    output_csv: str | None = None,
    num_samples: int | None = None,
    test_ids: list | None = None,
) -> pd.DataFrame:
    """
    Build a pass-through CSV for the description step that includes every
    image in the dataset, bypassing classification entirely.

    All rows are marked is_math=True so run_description() processes them
    without filtering.

    Args:
        output_csv: Destination path. Defaults to config.ALL_IMAGES_CSV.
        num_samples: Limit to first N rows (useful for testing).
        test_ids: Only include specific image IDs (useful for spot-testing).

    Returns:
        DataFrame with columns: image_id, question, is_math, final_category.
    """
    output_csv = output_csv or config.ALL_IMAGES_CSV

    df = load_mathvision()
    if test_ids:
        df = df[df["id"].astype(str).isin([str(i) for i in test_ids])]
    elif num_samples:
        df = df.head(num_samples)

    result = pd.DataFrame({
        "image_id":       df["id"].values,
        "question":       df["question"].values,
        "is_math":        True,
        "final_category": df["subject"].fillna("unknown").values,
    })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Prepared {len(result)} images → {output_csv}")
    return result
