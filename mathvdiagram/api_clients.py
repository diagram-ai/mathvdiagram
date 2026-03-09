import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic

from . import config


def get_openai_client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=config.OPENAI_API_KEY)


def get_gemini_model():
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    return genai.GenerativeModel(config.GEMINI_MODEL)


def get_claude_client() -> Anthropic:
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=config.ANTHROPIC_API_KEY)


def get_qwen_client() -> OpenAI:
    """Get Qwen client via OpenRouter (OpenAI-compatible API)."""
    if not config.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set. Set it in .env or environment.")
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_API_BASE,
    )


