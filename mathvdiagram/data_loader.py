"""
Data loader for the MathVision dataset from HuggingFace.

Loads the dataset once and provides access to images and metadata.
Images are cached locally after first download by the `datasets` library.
"""

import io
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
