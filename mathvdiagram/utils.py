import base64
import time
import os

import pandas as pd


def encode_image_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def call_with_retry(func, max_retries=5, initial_delay=1, max_delay=60):
    """Retry wrapper with exponential backoff for API calls."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            error_type = type(e).__name__

            is_quota_exhausted = any(
                x in error_str
                for x in ["resource_exhausted", "quota", "rpd", "daily quota", "daily limit"]
            )
            if is_quota_exhausted:
                print(f"   QUOTA EXHAUSTED: {str(e)[:200]}")
                return f"[QUOTA EXHAUSTED: {str(e)[:200]}]"

            is_rate_limit = "429" in error_str or "rate limit" in error_str
            is_timeout = any(
                x in error_str for x in ["504", "deadline", "timeout", "timed out"]
            )
            is_cancelled = "499" in error_str or "cancelled" in error_str or "canceled" in error_str
            is_server_error = any(
                x in error_str for x in ["500", "502", "503", "internal error", "service unavailable"]
            )

            if is_rate_limit:
                delay = min(initial_delay * (2 ** attempt), max_delay)
                print(f"   Rate limit hit. Waiting {delay}s (retry {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            elif is_timeout or is_cancelled or is_server_error:
                delay = min(initial_delay * (1.5 ** attempt), 30)
                label = "Timeout" if is_timeout else "Cancelled" if is_cancelled else "Server error"
                print(f"   {label}. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"   Non-retryable error ({error_type}): {str(e)[:100]}")
                return f"[API ERROR: {error_type} - {str(e)[:200]}]"

    return f"[API ERROR after {max_retries} retries: {str(last_error)[:200]}]"


def load_checkpoint(csv_path: str):
    """Load existing results from a checkpoint CSV. Returns (list[dict], set of processed ids)."""
    if not os.path.exists(csv_path):
        return [], set()
    try:
        df = pd.read_csv(csv_path)
        records = df.to_dict("records")
        ids = set(df["image_id"].tolist())
        return records, ids
    except Exception as e:
        print(f"Warning: Could not load checkpoint {csv_path}: {e}")
        return [], set()


def save_checkpoint(records: list, csv_path: str):
    """Save a list of dicts to CSV."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    pd.DataFrame(records).to_csv(csv_path, index=False)
