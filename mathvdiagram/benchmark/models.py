"""
models.py -- Model registry and API helpers for diagram generation.

Defines the set of models benchmarked, plus the OpenAI-compatible client and
the two generation pathways:

  * code_llm  -- returns code (TikZ / SVG / Python) that is compiled to a PNG
  * image_gen -- returns an image directly

Every provider is accessed through an OpenAI-compatible endpoint, so a single
client implementation covers all of them. API keys are read from the
environment (loaded from .env via ``mathvdiagram.benchmark``).
"""

import os

from openai import OpenAI

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    # --- DeepSeek (DEEPSEEK_API_KEY) ---
    "deepseek-v3": {
        "type": "code_llm",
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "deepseek-r1": {
        "type": "code_llm",
        "provider": "deepseek",
        "model_id": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "reasoning": True,
    },
    # --- OpenAI (OPENAI_API_KEY) ---
    "gpt-5.4": {
        "type": "code_llm",
        "provider": "openai",
        "model_id": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "reasoning": True,
    },
    "gpt-oss": {
        "type": "code_llm",
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    # --- OpenRouter (OPENROUTER_API_KEY) ---
    "claude-opus-4.6": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "anthropic/claude-opus-4.6",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "gemini-3.1-pro": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "google/gemini-3.1-pro-preview",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "qwen3.5-35b": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "qwen/qwen3.5-35b-a3b",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "max_tokens": 16384,
    },
    "llama-4-maverick": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "meta-llama/llama-4-maverick",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "kimi-k2.5": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "moonshotai/kimi-k2.5",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    # --- Image generation models (OpenRouter) ---
    "nano-banana-2": {
        "type": "image_gen",
        "provider": "openrouter",
        "model_id": "google/gemini-3.1-flash-image-preview",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "nano-banana-pro": {
        "type": "image_gen",
        "provider": "openrouter",
        "model_id": "google/gemini-3-pro-image-preview",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
}

# Human-readable display names, used by comparison reports and paper figures.
MODEL_NAMES = {
    "deepseek-v3": "DeepSeek V3",
    "deepseek-r1": "DeepSeek R1",
    "gpt-5.4": "GPT-5.4",
    "gpt-oss": "GPT-OSS-120B",
    "claude-opus-4.6": "Claude Opus 4.6",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "qwen3.5-35b": "Qwen3.5-35B",
    "llama-4-maverick": "Llama 4 Maverick",
    "kimi-k2.5": "Kimi K2.5",
    "nano-banana-2": "Nano Banana 2",
    "nano-banana-pro": "Nano Banana Pro",
}

# Canonical ordering for tables and figures.
MODEL_ORDER = list(MODEL_NAMES.keys())

SYSTEM_PROMPT = """You are an expert at generating precise mathematical diagrams.
Given a description, produce code that renders the described diagram as an image.
Choose whichever format you believe will produce the best result: TikZ (LaTeX), SVG, Python (matplotlib), or any other approach.
Output ONLY the code inside a single code block. No explanation, no commentary."""


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def get_client(model_cfg: dict) -> OpenAI:
    """Create an OpenAI-compatible client for the given model config."""
    api_key = os.environ.get(model_cfg["env_key"])
    if not api_key:
        raise ValueError(f"Set {model_cfg['env_key']} environment variable")

    # Image-gen models need longer timeouts for generation.
    timeout = 180 if model_cfg.get("type") == "image_gen" else 60
    return OpenAI(
        api_key=api_key,
        base_url=model_cfg["base_url"],
        timeout=timeout,
    )


def generate_code(client: OpenAI, model_cfg: dict, prompt: str) -> str:
    """Send prompt to a code LLM and return the raw text response."""
    params = {
        "model": model_cfg["model_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    if model_cfg.get("reasoning"):
        # Reasoning models: no temperature, use max_completion_tokens.
        params["max_completion_tokens"] = model_cfg.get("max_tokens", 16384)
    else:
        params["temperature"] = 0
        params["max_tokens"] = model_cfg.get("max_tokens", 8192)

    # Extra provider-specific params.
    params.update(model_cfg.get("extra_params", {}))

    response = client.chat.completions.create(**params)
    msg = response.choices[0].message
    # Try content first, then reasoning_content (DeepSeek-R1), then reasoning (OpenRouter).
    result = msg.content or getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    return result or ""


def generate_image(client: OpenAI, model_cfg: dict, prompt: str) -> bytes:
    """Send prompt to an image-gen model and return decoded image bytes."""
    response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[
            {"role": "user", "content": f"Generate a precise mathematical diagram: {prompt}"},
        ],
        extra_body={"modalities": ["image"]},
    )
    msg = response.choices[0].message
    image_data = None

    # Format 1: msg.images[0].image_url.url (OpenRouter native).
    images = getattr(msg, "images", None)
    if images:
        try:
            img_obj = images[0]
            if hasattr(img_obj, "image_url"):
                url = img_obj.image_url.url if hasattr(img_obj.image_url, "url") else ""
            elif isinstance(img_obj, dict):
                url = img_obj.get("image_url", {}).get("url", "")
            else:
                url = ""
            if url and "base64," in url:
                image_data = url.split("base64,", 1)[1]
        except (AttributeError, TypeError, IndexError):
            pass

    # Format 2: content is a list with image parts.
    if not image_data and isinstance(msg.content, list):
        for part in msg.content:
            try:
                if hasattr(part, "type") and part.type == "image_url":
                    url = part.image_url.url if hasattr(part.image_url, "url") else ""
                    if "base64," in url:
                        image_data = url.split("base64,", 1)[1]
                        break
            except (AttributeError, TypeError):
                continue

    # Format 3: content is a single base64 data URL string.
    if not image_data and isinstance(msg.content, str) and "base64," in msg.content:
        image_data = msg.content.split("base64,", 1)[1]

    if not image_data:
        raise ValueError(f"No image found in response. Content type: {type(msg.content)}")

    import base64
    return base64.b64decode(image_data)
