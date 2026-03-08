"""
Step 3: Consensus engine using Claude as an independent visual judge.

Takes the descriptions CSV (Gemini + OpenAI descriptions) and sends each image
along with both descriptions to Claude, which generates a consensus detailed
and concise prompt.
"""

import os

import pandas as pd
from tqdm import tqdm

from . import config
from .api_clients import get_claude_client
from .data_loader import load_mathvision, get_image_base64
from .utils import load_checkpoint, save_checkpoint


def _build_consensus_prompt(row: dict, has_gemini: bool) -> tuple[str, str]:
    """Build the system and user prompts for consensus generation."""
    problem = row["question"]
    desc_openai = row["openai_prompt"]

    if has_gemini:
        desc_gemini = row["gemini_prompt"]
        descriptions_section = f"""
--- Description A (from Gemini) ---
{desc_gemini}

--- Description B (from OpenAI GPT-4) ---
{desc_openai}
"""
        task_instruction = """
Your Task:
1. LOOK at the image carefully.
2. Read both AI descriptions (Gemini and OpenAI).
3. Compare YOUR visual understanding with both descriptions.
4. Identify where descriptions AGREE with the image.
5. Identify CONFLICTS (where descriptions disagree or miss details you see).
6. Use the problem context as tie-breaker if needed.
7. Generate a FINAL, authoritative description that a blind illustrator could use to perfectly recreate this diagram.
"""
    else:
        descriptions_section = f"""
--- Description (from OpenAI GPT-4) ---
{desc_openai}

NOTE: Gemini description unavailable (error in Phase 1).
"""
        task_instruction = """
Your Task:
1. LOOK at the image carefully.
2. Read the OpenAI description.
3. Compare YOUR visual understanding with the OpenAI description.
4. Identify what OpenAI captured correctly.
5. Identify what OpenAI missed or got wrong based on what you see.
6. Use the problem context for additional guidance.
7. Generate a FINAL, authoritative description that a blind illustrator could use to perfectly recreate this diagram.
"""

    system_prompt = f"""
You are an INDEPENDENT VISUAL JUDGE for a mathematical diagram benchmarking dataset.

You will be given:
1. The actual IMAGE (you can see it)
2. The original math problem (context)
3. AI-generated descriptions from other models

{task_instruction}

You must generate TWO versions of the consensus prompt:
1. DETAILED: Extremely detailed, comprehensive visual description (like for a blind illustrator)
2. CONCISE: Short 3-4 line summary capturing the essence

Output format (STRICTLY follow):
[CONFLICT_SCORE]: <Low/Medium/High>
[RATIONALE]: <Brief explanation of what you saw vs. what descriptions said>
[DETAILED_PROMPT]: <Extremely detailed, comprehensive visual instruction with all specifics>
[CONCISE_PROMPT]: <3-4 line concise summary of the diagram>
"""

    user_content = f"""
--- Original Problem Context ---
{problem}

{descriptions_section}

Now look at the image above and generate the consensus description.
"""
    return system_prompt, user_content


def generate_consensus_for_row(claude_client, row: dict) -> str:
    """Send image + descriptions to Claude and get consensus output."""
    image_id = row["image_id"]
    has_gemini = not (
        pd.isna(row.get("gemini_prompt"))
        or "ERROR" in str(row.get("gemini_prompt", "")).upper()
    )

    base64_image, media_type_or_error = get_image_base64(image_id)
    if base64_image is None:
        return f"[IMAGE_ERROR: {media_type_or_error}]"

    system_prompt, user_content = _build_consensus_prompt(row, has_gemini)

    try:
        message = claude_client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type_or_error,
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": system_prompt + "\n\n" + user_content},
                    ],
                }
            ],
            temperature=0.2,
        )
        return message.content[0].text
    except Exception as e:
        return f"[CLAUDE_ERROR: {str(e)[:200]}]"


def _parse_consensus_output(output: str) -> tuple[str, str]:
    """Extract detailed and concise prompts from Claude's output."""
    if "[DETAILED_PROMPT]:" in output:
        detailed = output.split("[DETAILED_PROMPT]:")[1].split("[CONCISE_PROMPT]")[0].strip()
        concise = (
            output.split("[CONCISE_PROMPT]:")[1].strip()
            if "[CONCISE_PROMPT]:" in output
            else "PARSING_ERROR"
        )
    else:
        detailed = "PARSING_ERROR"
        concise = "PARSING_ERROR"
    return detailed, concise


def _is_retryable_error(value: str) -> bool:
    error_keywords = [
        "insufficient_quota", "rate_limit", "429", "quota",
        "credit", "timeout", "overloaded",
    ]
    return any(kw in str(value).lower() for kw in error_keywords)


def run_consensus(
    input_csv: str | None = None,
    resume: bool = True,
):
    """
    Run the consensus pipeline: Claude judges each image against descriptions.

    Args:
        input_csv: Path to descriptions CSV. Defaults to config.DESCRIPTIONS_CSV.
        resume: Resume from existing checkpoint.

    Returns:
        DataFrame with consensus prompts.
    """
    input_csv = input_csv or config.DESCRIPTIONS_CSV
    output_csv = config.CONSENSUS_CSV

    # Ensure HF dataset is loaded so images are available
    load_mathvision()

    print(f"Loading descriptions from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"ERROR: {input_csv} not found. Run the describe step first.")
        return None
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    # Load existing results
    existing_results, processed_ids = load_checkpoint(output_csv) if resume else ([], set())
    retry_ids = set()

    if existing_results and resume:
        existing_df = pd.DataFrame(existing_results)
        # Identify successful vs retryable
        successful_mask = (
            existing_df["detailed_prompt"].notna()
            & (existing_df["detailed_prompt"] != "PARSING_ERROR")
            & ~existing_df["detailed_prompt"].apply(_is_retryable_error)
            & existing_df["concise_prompt"].notna()
            & (existing_df["concise_prompt"] != "PARSING_ERROR")
        )
        processed_ids = set(existing_df.loc[successful_mask, "image_id"].values)
        retry_ids = set(
            existing_df.loc[
                existing_df["detailed_prompt"].apply(_is_retryable_error)
                | existing_df["concise_prompt"].apply(_is_retryable_error),
                "image_id",
            ].values
        )
        print(f"Resuming: {len(processed_ids)} done, {len(retry_ids)} to retry")

    claude_client = get_claude_client()
    results = []
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Consensus"):
        image_id = row["image_id"]

        if image_id in processed_ids and image_id not in retry_ids:
            # Use cached result
            cached = [r for r in existing_results if r["image_id"] == image_id]
            if cached:
                results.append(cached[0])
            continue

        try:
            claude_output = generate_consensus_for_row(claude_client, row.to_dict())
            detailed, concise = _parse_consensus_output(claude_output)

            result_row = row.to_dict()
            result_row["claude_consensus_output"] = claude_output
            result_row["detailed_prompt"] = detailed
            result_row["concise_prompt"] = concise
            results.append(result_row)

            if "ERROR" in claude_output or "PARSING_ERROR" in detailed:
                error_count += 1
        except Exception as e:
            result_row = row.to_dict()
            result_row["claude_consensus_output"] = f"[EXCEPTION: {str(e)}]"
            result_row["detailed_prompt"] = f"[ERROR: {str(e)[:100]}]"
            result_row["concise_prompt"] = f"[ERROR: {str(e)[:100]}]"
            results.append(result_row)
            error_count += 1

        if len(results) % config.CHECKPOINT_EVERY == 0 and results:
            _merge_and_save(results, existing_results, output_csv)

    _merge_and_save(results, existing_results, output_csv)
    final_df = pd.DataFrame(results)
    print(f"\nConsensus complete: {len(final_df)} images, {error_count} errors")
    return final_df


def _merge_and_save(new_results: list, existing_results: list, output_csv: str):
    """Merge new results with existing and save."""
    new_df = pd.DataFrame(new_results)
    if existing_results:
        existing_df = pd.DataFrame(existing_results)
        new_ids = set(new_df["image_id"].values)
        keep = existing_df[~existing_df["image_id"].isin(new_ids)]
        merged = pd.concat([keep, new_df], ignore_index=True).sort_values("image_id")
    else:
        merged = new_df
    save_checkpoint(merged.to_dict("records"), output_csv)


if __name__ == "__main__":
    run_consensus()
