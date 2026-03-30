"""
Step 3: Aggregate descriptions using Qwen (open-source VLM via OpenRouter).

Takes the 3 proprietary descriptions (OpenAI, Gemini, Claude) and the original image,
sends them to Qwen which produces a final authoritative description by:
1. Identifying what all 3 descriptions agree on
2. Resolving conflicts by checking the image
3. Adding anything all 3 missed
"""

import os
import time
import logging

import pandas as pd
from tqdm import tqdm

from . import config
from .api_clients import get_qwen_client, get_llama_client
from .data_loader import load_mathvision, get_image_base64
from .utils import call_with_retry, load_checkpoint, save_checkpoint


logger = logging.getLogger(__name__)


AGGREGATION_PROMPT = """You are an expert mathematical diagram analyst performing description aggregation.

You have been given:
1. The ORIGINAL IMAGE (you can see it)
2. Independent descriptions of this image from different AI models
3. The original math problem (for context only — do NOT solve it)

Your task is to produce a SINGLE, FINAL, AUTHORITATIVE description by following this process:

STEP 1 — AGREEMENT ANALYSIS:
Identify visual elements that the descriptions agree on. These are high-confidence elements — include them.

STEP 2 — CONFLICT RESOLUTION:
Where descriptions DISAGREE (e.g., one says "right angle at B" and another says "right angle at C"), LOOK AT THE IMAGE and determine which is correct. State what you see.

STEP 3 — COVERAGE CHECK:
Look at the image yourself. Is there ANYTHING visible that NONE of the descriptions mention? Small labels, faint marks, background elements, line styles? Add anything they all missed.

STEP 4 — WRITE THE FINAL DESCRIPTION:
Produce a comprehensive, precise description that a professional illustrator who cannot see the image could use to recreate it perfectly. Include:
- Every shape, line, and curve with its exact style (solid/dashed/dotted, thick/thin)
- Every label, number, and text element transcribed exactly
- Every geometric notation (angle marks, congruence ticks, parallel arrows, right-angle squares)
- Spatial layout (what is where relative to what)
- Colors and shading
- Scale and proportions

OUTPUT FORMAT:
[AGREEMENT]: Brief summary of what descriptions agree on (2-3 sentences)
[CONFLICTS]: Any conflicts found and how you resolved them. Write "None" if no conflicts.
[ADDITIONS]: Anything you spotted that all descriptions missed. Write "None" if nothing.
[FINAL_DESCRIPTION]: The complete, authoritative description. This is the most important part.
"""


def _is_valid_description(text) -> bool:
    """Check if a description is usable (not an error message)."""
    if not text or not isinstance(text, str):
        return False
    if pd.isna(text):
        return False
    if len(text.strip()) < 50:
        return False
    error_markers = ["[ERROR", "[OPENAI ERROR", "[GEMINI ERROR", "[CLAUDE ERROR", "[QUOTA", "nan"]
    return not any(marker in text[:30] for marker in error_markers)


def _build_aggregation_message(row: dict, image_b64: str, media_type: str) -> list[dict]:
    """Build the message payload for Qwen aggregation."""
    question = row.get("question", "")
    category = row.get("category", "unknown")

    # Collect available descriptions — support both old and new column names
    descriptions = []
    provider_columns = [
        ("openai", "description_openai", "openai_prompt"),
        ("gemini", "description_gemini", "gemini_prompt"),
        ("claude", "description_claude", None),
    ]

    for provider_name, new_col, old_col in provider_columns:
        text = str(row.get(new_col, ""))
        if not _is_valid_description(text) and old_col:
            text = str(row.get(old_col, ""))
        if _is_valid_description(text):
            descriptions.append(f"--- Description from {provider_name.upper()} ---\n{text}")

    if not descriptions:
        return []

    descriptions_text = "\n\n".join(descriptions)

    user_text = f"""{AGGREGATION_PROMPT}

Image category: {category}
Question context (do NOT solve): {question}
Number of descriptions available: {len(descriptions)}

{descriptions_text}

Now look at the image and produce the aggregated output."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                },
            ],
        },
    ]


def aggregate_single_image(qwen_client, row: dict, image_b64: str, media_type: str) -> str:
    """Send image + descriptions to Qwen and get aggregated output."""
    messages = _build_aggregation_message(row, image_b64, media_type)
    if not messages:
        return "[AGGREGATION_ERROR: No valid descriptions available]"

    def _call():
        response = qwen_client.chat.completions.create(
            model=config.QWEN_MODEL,
            messages=messages,
            max_tokens=config.AGGREGATION_MAX_TOKENS,
            temperature=0.2,
        )
        return response.choices[0].message.content

    start = time.time()
    result = call_with_retry(_call, max_retries=3)
    elapsed_ms = (time.time() - start) * 1000
    response_len = len(result) if isinstance(result, str) else 0
    logger.info(
        "[TIMING] qwen.aggregate response_ms=%.1f response_len=%d",
        elapsed_ms,
        response_len,
    )
    if isinstance(result, str) and "[API ERROR" in result:
        return f"[QWEN_ERROR: {result}]"
    return result


def _parse_aggregated_output(output: str) -> dict:
    """Parse Qwen's structured output into components."""
    result = {
        "agreement": "",
        "conflicts": "",
        "additions": "",
        "final_description": "",
        "raw_output": output,
    }

    if not output or not isinstance(output, str) or "ERROR" in output[:20]:
        return result

    sections = {
        "[AGREEMENT]": "agreement",
        "[CONFLICTS]": "conflicts",
        "[ADDITIONS]": "additions",
        "[FINAL_DESCRIPTION]": "final_description",
    }

    for marker, key in sections.items():
        search = marker + ":"
        if search in output:
            start = output.index(search) + len(search)
            # Find next section marker
            end = len(output)
            for next_marker in sections:
                next_search = next_marker + ":"
                if next_marker != marker and next_search in output:
                    next_pos = output.index(next_search)
                    if next_pos > start and next_pos < end:
                        end = next_pos
            result[key] = output[start:end].strip()

    # Fallback: if parsing failed, use the whole output
    if not result["final_description"] and len(output) > 100:
        result["final_description"] = output

    return result


def run_consensus(
    input_csv: str | None = None,
    resume: bool = True,
    delay: float | None = None,
):
    """
    Run Qwen aggregation over the 3 proprietary descriptions.

    This replaces the old Claude consensus step. The function name
    is kept as run_consensus for backward compatibility with pipeline.py.

    Args:
        input_csv: Path to descriptions CSV. Defaults to config.DESCRIPTIONS_CSV.
        resume: Resume from existing checkpoint.
        delay: Seconds between API calls.

    Returns:
        DataFrame with aggregated descriptions.
    """
    input_csv = input_csv or config.DESCRIPTIONS_CSV
    output_csv = config.AGGREGATED_CSV
    delay = delay if delay is not None else config.DELAY_BETWEEN_REQUESTS

    # Ensure HF dataset is loaded for images
    load_mathvision()

    print(f"Loading descriptions from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"ERROR: {input_csv} not found. Run the describe step first.")
        return None
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    # Count valid descriptions per provider
    for new_col, old_col in [("description_openai", "openai_prompt"), ("description_gemini", "gemini_prompt"), ("description_claude", None)]:
        col = new_col if new_col in df.columns else old_col
        if col and col in df.columns:
            valid = df[col].apply(lambda x: _is_valid_description(str(x))).sum()
            print(f"  {col}: {valid} valid")

    # Load checkpoint
    existing_results, processed_ids = load_checkpoint(output_csv) if resume else ([], set())
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} already aggregated")

    # Initialize Qwen client
    qwen_client = get_qwen_client()

    results = list(existing_results)
    error_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating (Qwen)"):
        image_id = row["image_id"]
        if image_id in processed_ids:
            continue

        # Get image
        image_b64, media_type = get_image_base64(image_id)
        if image_b64 is None:
            result_row = row.to_dict()
            result_row["aggregation_raw"] = f"[IMAGE_ERROR: {media_type}]"
            result_row["agreement"] = ""
            result_row["conflicts"] = ""
            result_row["additions"] = ""
            result_row["final_description"] = ""
            result_row["n_descriptions_used"] = 0
            results.append(result_row)
            processed_ids.add(image_id)
            error_count += 1
            continue

        # Count valid descriptions
        n_valid = 0
        for new_col, old_col in [("description_openai", "openai_prompt"), ("description_gemini", "gemini_prompt"), ("description_claude", None)]:
            col = new_col if new_col in row.index else old_col
            if col and _is_valid_description(str(row.get(col, ""))):
                n_valid += 1

        if n_valid == 0:
            result_row = row.to_dict()
            result_row["aggregation_raw"] = "[NO_DESCRIPTIONS: All providers failed]"
            result_row["agreement"] = ""
            result_row["conflicts"] = ""
            result_row["additions"] = ""
            result_row["final_description"] = ""
            result_row["n_descriptions_used"] = 0
            results.append(result_row)
            processed_ids.add(image_id)
            error_count += 1
            continue

        time.sleep(delay)

        try:
            raw_output = aggregate_single_image(
                qwen_client, row.to_dict(), image_b64, media_type,
            )
            parsed = _parse_aggregated_output(raw_output)

            result_row = row.to_dict()
            result_row["aggregation_raw"] = raw_output
            result_row["agreement"] = parsed["agreement"]
            result_row["conflicts"] = parsed["conflicts"]
            result_row["additions"] = parsed["additions"]
            result_row["final_description"] = parsed["final_description"]
            result_row["n_descriptions_used"] = n_valid
            results.append(result_row)

            if "ERROR" in str(raw_output)[:20]:
                error_count += 1

        except Exception as e:
            result_row = row.to_dict()
            result_row["aggregation_raw"] = f"[EXCEPTION: {str(e)[:200]}]"
            result_row["agreement"] = ""
            result_row["conflicts"] = ""
            result_row["additions"] = ""
            result_row["final_description"] = ""
            result_row["n_descriptions_used"] = n_valid
            results.append(result_row)
            error_count += 1

        processed_ids.add(image_id)

        if len(results) % config.CHECKPOINT_EVERY == 0 and results:
            save_checkpoint(results, output_csv)
            print(f"  Checkpoint: {len(results)} images")

    # Final save
    save_checkpoint(results, output_csv)
    result_df = pd.DataFrame(results)

    # Print summary
    has_final = result_df["final_description"].apply(
        lambda x: isinstance(x, str) and len(str(x)) > 50
    ).sum()

    print(f"\nAggregation complete:")
    print(f"  Total images: {len(result_df)}")
    print(f"  Successful: {has_final}")
    print(f"  Errors: {error_count}")

    if "n_descriptions_used" in result_df.columns:
        print(f"  Descriptions used per image:")
        for n, count in result_df["n_descriptions_used"].value_counts().sort_index().items():
            print(f"    {n} providers: {count} images")

    if "conflicts" in result_df.columns:
        has_conflicts = result_df["conflicts"].apply(
            lambda x: isinstance(x, str) and len(str(x)) > 5 and str(x).strip().lower() != "none"
        ).sum()
        print(f"  Images with conflicts resolved: {has_conflicts}")

    if "additions" in result_df.columns:
        has_additions = result_df["additions"].apply(
            lambda x: isinstance(x, str) and len(str(x)) > 5 and str(x).strip().lower() != "none"
        ).sum()
        print(f"  Images with missed elements added: {has_additions}")

    return result_df


# Backward compatibility alias
run_aggregation = run_consensus


# ── Benchmarking pipeline: Llama 3.3-70B prompt synthesis ────────────────────

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a visual reconstruction specialist. Synthesize multiple diagram descriptions
into one concise reconstruction prompt for a diagram-generation agent.

Priority order — include what matters most first:
1. Overall structure: what type of diagram is it, how many main elements
2. Labels and numbers: exact text, vertex names, axis values, data values, counts
3. Spatial layout: what is positioned where relative to what

Skip entirely unless they carry specific meaning:
- Line thickness or style (solid/dashed is fine to skip unless dashes are significant)
- White background, absent grid lines, no shading (these are default assumptions)
- Generic styling that doesn't affect diagram identity

Format:
- Start with "Draw ..."
- 3-5 sentences, maximum 120 words
- No meta-commentary ("The descriptions agree...", "According to model X...")\
"""

_SYNTHESIS_PROVIDER_COLUMNS = [
    ("OpenAI", "description_openai"),
    ("Gemini", "description_gemini"),
    ("Claude", "description_claude"),
    ("Llama",  "description_llama"),
]


def _collect_valid_descriptions(row: dict, provider_columns: list | None = None) -> list[str]:
    """Return formatted description strings for providers that succeeded."""
    out = []
    for name, col in (provider_columns or _SYNTHESIS_PROVIDER_COLUMNS):
        text = str(row.get(col, "")).strip()
        if len(text) > 50 and not text[:30].startswith(
            ("[ERROR", "[OPENAI", "[GEMINI", "[CLAUDE", "[QWEN", "[LLAMA", "[API", "[QUOTA")
        ):
            out.append(f"[{name}]:\n{text}")
    return out


def synthesize_single_image(llama_client, row: dict, provider_columns: list | None = None, min_providers: int | None = None) -> str:
    """Use Llama 3.3-70B to produce a concise reconstruction prompt from descriptions."""
    cols = provider_columns or _SYNTHESIS_PROVIDER_COLUMNS
    required = min_providers if min_providers is not None else len(cols)
    descriptions = _collect_valid_descriptions(row, provider_columns=cols)
    if len(descriptions) < required:
        missing = required - len(descriptions)
        return f"[SYNTHESIS_SKIPPED: Only {len(descriptions)}/{required} providers available — {missing} missing]"

    user_message = (
        f"Here are {len(descriptions)} independent visual descriptions of the same diagram:\n\n"
        + "\n\n".join(descriptions)
        + f"\n\nQuestion context (do NOT solve): {row.get('question', '')}\n\n"
        "Synthesize these into one concise reconstruction prompt starting with \"Draw\"."
    )

    def _call():
        response = llama_client.chat.completions.create(
            model=config.LLAMA_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=config.PROMPT_SYNTH_MAX_TOKENS,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    start = time.time()
    result = call_with_retry(_call, max_retries=3)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "[TIMING] llama_judge.synthesize response_ms=%.1f response_len=%d",
        elapsed_ms,
        len(result) if isinstance(result, str) else 0,
    )
    return result


def run_prompt_synthesis(
    input_csv: str | None = None,
    output_csv: str | None = None,
    resume: bool = True,
    delay: float | None = None,
    provider_columns: list | None = None,
    min_providers: int | None = None,
) -> pd.DataFrame:
    """
    Synthesize a concise visual reconstruction prompt for every image using
    Llama 3.3-70B (text-only, via Groq) as the judge.

    Resumable — skips already-processed image IDs on restart.

    Args:
        input_csv: Path to descriptions CSV. Defaults to config.DESCRIPTIONS_CSV.
        output_csv: Destination CSV. Defaults to config.CONCISE_PROMPTS_CSV.
        resume: Resume from existing checkpoint.
        delay: Seconds between API calls.
        provider_columns: List of (name, col) tuples for active providers.
                          Defaults to _SYNTHESIS_PROVIDER_COLUMNS (all 4).
                          Pass a subset for DaTikZ or other 2-provider runs.

    Returns:
        DataFrame with columns: image_id, question, category,
        concise_prompt, n_descriptions_used.
    """
    input_csv        = input_csv        or config.DESCRIPTIONS_CSV
    output_csv       = output_csv       or config.CONCISE_PROMPTS_CSV
    delay            = delay if delay is not None else config.DELAY_BETWEEN_REQUESTS
    provider_columns = provider_columns or _SYNTHESIS_PROVIDER_COLUMNS

    if not os.path.exists(input_csv):
        print(f"ERROR: {input_csv} not found. Run the describe step first.")
        return None

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows for prompt synthesis (requires {len(provider_columns)} providers)")

    results, processed_ids = load_checkpoint(output_csv) if resume else ([], set())
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} already synthesized")

    llama_client = get_llama_client()
    error_count  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Synthesizing prompts (Llama 3.3-70B)"):
        image_id = row["image_id"]
        if image_id in processed_ids:
            continue

        time.sleep(delay)
        row_dict = row.to_dict()
        n_valid = len(_collect_valid_descriptions(row_dict, provider_columns=provider_columns))

        try:
            concise_prompt = synthesize_single_image(llama_client, row_dict, provider_columns=provider_columns, min_providers=min_providers)
            if str(concise_prompt)[:5].startswith("["):
                error_count += 1
        except Exception as e:
            concise_prompt = f"[EXCEPTION: {str(e)[:200]}]"
            error_count += 1

        results.append({
            "image_id":            image_id,
            "question":            row.get("question", ""),
            "category":            row.get("category", ""),
            "concise_prompt":      concise_prompt,
            "n_descriptions_used": n_valid,
        })
        processed_ids.add(image_id)

        if len(results) % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(results, output_csv)
            print(f"  Checkpoint: {len(results)} images")

    save_checkpoint(results, output_csv)
    result_df = pd.DataFrame(results)

    successful = result_df["concise_prompt"].apply(
        lambda x: isinstance(x, str) and len(x) > 20 and not str(x)[:5].startswith("[")
    ).sum()
    print(f"\nSynthesis complete:")
    print(f"  Total:      {len(result_df)}")
    print(f"  Successful: {successful}")
    print(f"  Errors:     {error_count}")
    return result_df


if __name__ == "__main__":
    run_consensus()
