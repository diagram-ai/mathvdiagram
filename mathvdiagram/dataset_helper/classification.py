"""
Structured LLM classification (Layer 2) using taxonomy-aware prompts.

Supports multiple providers: OpenAI, Gemini, and Claude.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .. import config
from ..api_clients import get_openai_client, get_gemini_model, get_claude_client
from ..data_loader import load_mathvision, get_image_base64
from ..utils import call_with_retry, load_checkpoint, save_checkpoint
from .taxonomy import VALID_CATEGORIES, build_taxonomy_text


STRUCTURED_CLASSIFICATION_PROMPT = f"""You are a mathematical diagram classifier. Your job is to analyze an image and classify it into exactly one of six categories.

TAXONOMY OF CATEGORIES:

{build_taxonomy_text()}

VALID CATEGORY IDs: {', '.join(VALID_CATEGORIES)}

INSTRUCTIONS:
1. Look at the image carefully.
2. Consider the question context provided.
3. Consider the MathVision subject tag provided.
4. Evaluate each boolean feature listed below.
5. Use the boolean checklist fields to guide your category decision. For example, if has_real_world_objects=true and has_geometric_labels=false, the category is almost certainly non_diagram.
6. Select exactly ONE category from the taxonomy above.
7. Respond with ONLY a valid JSON object. No markdown fences, no preamble, no explanation outside the JSON.

Respond with this exact JSON structure:
{{"has_labeled_axes": true/false, "has_geometric_labels": true/false, "has_real_world_objects": true/false, "has_photographic_content": true/false, "has_mathematical_notation": true/false, "has_grid_or_coordinate_system": true/false, "diagram_category": "<one of the 6 category IDs>", "confidence": "high/medium/low", "reasoning": "<1-2 sentence explanation>"}}

CRITICAL RULES:
- If the image contains real-world objects (animals, flowers, trains, buildings, people) used to present a math problem, classify as "non_diagram" regardless of whether numbers or math operations appear.
- "geometric_construction" requires formal vertex labels (A, B, C) or angle marks on abstract line-drawn shapes.
- "coordinate_plot" requires explicit labeled axes.
- Output ONLY the JSON object. No other text."""


_PROVIDER_MODELS: dict[str, str] = {
    "openai": config.OPENAI_CLASSIFIER_MODEL,
    "gemini": config.GEMINI_MODEL,
    "claude": config.CLAUDE_MODEL,
}

_PROVIDER_CLIENTS: dict = {}


def _get_client(provider: str):
    """Get or cache the API client for a provider."""
    if provider not in _PROVIDER_CLIENTS:
        if provider == "openai":
            _PROVIDER_CLIENTS[provider] = get_openai_client()
        elif provider == "gemini":
            _PROVIDER_CLIENTS[provider] = get_gemini_model()
        elif provider == "claude":
            _PROVIDER_CLIENTS[provider] = get_claude_client()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _PROVIDER_CLIENTS[provider]


def parse_classification_response(raw_response: str) -> dict | None:
    """
    Parse the JSON classification response from the LLM.
    Handles markdown fences, trailing commas, partial JSON, single quotes.
    """
    if not raw_response or not isinstance(raw_response, str):
        return None

    text = raw_response.strip()

    # Remove markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    text = text[start:end + 1]

    # Remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Try parsing as-is first
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try replacing single quotes with double quotes
        try:
            fixed = text.replace("'", '"')
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            print(f"  Warning: could not parse classification JSON")
            return None

    if isinstance(parsed, dict) and "diagram_category" in parsed:
        if parsed["diagram_category"] not in VALID_CATEGORIES:
            parsed["diagram_category"] = "non_diagram"
            parsed["confidence"] = "low"
        return parsed
    return None


def classify_single_image_structured(
    client,
    image_b64: str,
    question_text: str,
    subject: str,
    model: str | None = None,
    provider: str = "openai",
) -> dict:
    """
    Send one image to the LLM with the structured classification prompt.

    Supports providers: "openai", "gemini", "claude".
    Returns parsed dict, or an error dict on failure.
    """
    model = model or _PROVIDER_MODELS.get(provider, config.OPENAI_CLASSIFIER_MODEL)
    user_context = f"Subject: {subject}\nThe associated math problem is: {question_text}\n\nClassify this image:"

    def _call_openai():
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STRUCTURED_CLASSIFICATION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_context},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_completion_tokens=300,
            timeout=30,
        )
        return response.choices[0].message.content

    def _call_gemini():
        # Convert base64 to PIL for Gemini
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        prompt = f"{STRUCTURED_CLASSIFICATION_PROMPT}\n\n{user_context}"
        response = client.generate_content([prompt, pil_image])
        return response.text

    def _call_claude():
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{STRUCTURED_CLASSIFICATION_PROMPT}\n\n{user_context}"},
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

    call_fn = {"openai": _call_openai, "gemini": _call_gemini, "claude": _call_claude}
    if provider not in call_fn:
        return {"error": f"Unknown provider: {provider}", "diagram_category": "unknown", "confidence": "low"}

    raw = call_with_retry(call_fn[provider], max_retries=3)
    if isinstance(raw, str) and "[API ERROR" in raw:
        return {"error": raw, "raw_response": raw, "diagram_category": "unknown", "confidence": "low"}

    parsed = parse_classification_response(raw)
    if parsed is None:
        return {"error": "Failed to parse response", "raw_response": raw, "diagram_category": "unknown", "confidence": "low"}
    return parsed


def classify_batch_structured(
    df: pd.DataFrame,
    resume: bool = True,
    delay: float | None = None,
    providers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Classify a batch of images using structured prompts across multiple providers.

    For each provider, stores:
      - cls_{provider}: full JSON classification dict (as string)
      - cat_{provider}: extracted diagram_category
      - conf_{provider}: extracted confidence

    Supports checkpointing.
    """
    delay = delay if delay is not None else config.DELAY_BETWEEN_REQUESTS
    providers = providers or ["openai"]  # Single provider is sufficient
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "structured_classification.csv")

    load_mathvision()

    results, processed_ids = load_checkpoint(checkpoint_path) if resume else ([], set())
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} already classified")

    # Pre-initialize clients
    clients = {}
    for prov in providers:
        try:
            clients[prov] = _get_client(prov)
        except (ValueError, Exception) as e:
            print(f"  Warning: could not init {prov} client: {e}")

    active_providers = [p for p in providers if p in clients]
    if not active_providers:
        print("No providers available. Check API keys.")
        return pd.DataFrame(results) if results else df.copy()

    processed_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Structured classification"):
        image_id = str(row.get("id", row.get("image_id", "")))
        if image_id in processed_ids:
            continue

        b64, _ = get_image_base64(image_id)
        if b64 is None:
            entry = {
                "image_id": image_id,
                "question": row.get("question", ""),
                "subject": row.get("subject", ""),
                "level": row.get("level", ""),
            }
            for prov in active_providers:
                err = json.dumps({"error": "image_not_found", "diagram_category": "unknown", "confidence": "low"})
                entry[f"cls_{prov}"] = err
                entry[f"cat_{prov}"] = "unknown"
                entry[f"conf_{prov}"] = "low"
            results.append(entry)
            processed_ids.add(image_id)
            processed_count += 1
            continue

        entry = {
            "image_id": image_id,
            "question": row.get("question", ""),
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
        }

        for prov in active_providers:
            time.sleep(delay)
            classification = classify_single_image_structured(
                clients[prov], b64, entry["question"], entry["subject"], provider=prov,
            )
            entry[f"cls_{prov}"] = json.dumps(classification)
            entry[f"cat_{prov}"] = classification.get("diagram_category", "unknown")
            entry[f"conf_{prov}"] = classification.get("confidence", "low")

        results.append(entry)
        processed_ids.add(image_id)
        processed_count += 1

        if processed_count % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(results, checkpoint_path)
            print(f"  Checkpoint: {processed_count} images")

    save_checkpoint(results, checkpoint_path)
    print(f"\nStructured classification complete: {len(results)} images across {active_providers}")
    return pd.DataFrame(results)
