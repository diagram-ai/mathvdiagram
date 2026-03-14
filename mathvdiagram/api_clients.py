from __future__ import annotations

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from anthropic import Anthropic  # type: ignore
except Exception:  # pragma: no cover
    Anthropic = None

from . import config


def get_openai_client():
    if OpenAI is None:
        raise ImportError("openai package is not installed. Install it with `pip install openai`.")
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=config.OPENAI_API_KEY)


def get_gemini_model():
    if genai is None:
        raise ImportError(
            "google-generativeai package is not installed. Install it with `pip install google-generativeai`."
        )
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    return genai.GenerativeModel(config.GEMINI_MODEL)


def get_claude_client():
    if Anthropic is None:
        raise ImportError("anthropic package is not installed. Install it with `pip install anthropic`.")
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=config.ANTHROPIC_API_KEY)


def get_qwen_client():
    """Get Qwen client via OpenRouter (OpenAI-compatible API)."""
    if OpenAI is None:
        raise ImportError("openai package is not installed. Install it with `pip install openai`.")
    if not config.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set. Set it in .env or environment.")
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_API_BASE,
    )


def get_llama_client():
    """Get Llama Vision client via Groq (OpenAI-compatible API)."""
    if OpenAI is None:
        raise ImportError("openai package is not installed. Install it with `pip install openai`.")
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Set it in .env for the Llama vision provider.")
    return OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_API_BASE,
    )


