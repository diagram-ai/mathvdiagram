"""
Main pipeline orchestrator.

Runs the full benchmarking pipeline:
  Step 1: Classify images — taxonomy-based multi-model (default) or legacy GPT binary
  Step 2: Generate descriptions from OpenAI + Gemini + Claude (3 independent VLMs)
  Step 3: Qwen aggregation (open-source VLM via OpenRouter) to produce final description
"""

import os
import argparse

import pandas as pd

from . import config
from .classify import run_classification
from .describe import run_description, retry_failed_descriptions
from .consensus import run_consensus, run_prompt_synthesis, _SYNTHESIS_PROVIDER_COLUMNS
from .report import generate_report, generate_benchmarking_report
from .data_loader import prepare_all_images, prepare_datikz_images, get_datikz_image_pil, get_datikz_image_base64
from .dataset_helper.pipeline import run_full_classification


def _generate_legacy_csvs(full_csv: str) -> None:
    """
    Read full_classification.csv and write legacy bridge files:
      - classification_results.csv  (is_diagram=True rows, with is_math=True)
      - skipped_non_math.csv        (is_diagram=False rows, with is_math=False)

    This lets describe.py, consensus.py, and report.py work unchanged.
    """
    df = pd.read_csv(full_csv)

    diagrams = df[df["is_diagram"] == True].copy()
    non_diagrams = df[df["is_diagram"] == False].copy()

    # Map to legacy column names
    diagrams["is_math"] = True
    non_diagrams["is_math"] = False
    non_diagrams["reason"] = "non_diagram"

    # Ensure image_id and question columns exist
    for part in (diagrams, non_diagrams):
        if "image_id" not in part.columns and "id" in part.columns:
            part["image_id"] = part["id"]
        if "question" not in part.columns:
            part["question"] = ""

    diagrams.to_csv(config.CLASSIFICATION_CSV, index=False)
    non_diagrams.to_csv(config.SKIPPED_CSV, index=False)
    print(f"  Legacy bridge: {len(diagrams)} math → {config.CLASSIFICATION_CSV}")
    print(f"  Legacy bridge: {len(non_diagrams)} skipped → {config.SKIPPED_CSV}")


def run_pipeline(
    num_samples: int | None = None,
    test_ids: list | None = None,
    resume: bool = True,
    delay: float | None = None,
    skip_classify: bool = False,
    skip_describe: bool = False,
    skip_aggregate: bool = False,
    providers: list[str] | None = None,
    reliability_threshold: float = 0.7,
    legacy_classify: bool = False,
):
    """
    Run the full pipeline from classification through aggregation.

    Args:
        num_samples: Limit to first N samples. None = all.
        test_ids: Only process specific IDs. None = all.
        resume: Resume from existing checkpoints.
        delay: Seconds between API calls.
        skip_classify: Skip step 1 if classification CSV already exists.
        skip_describe: Skip step 2 if descriptions CSV already exists.
        skip_aggregate: Skip step 3 if aggregated CSV already exists.
        providers: LLM providers for taxonomy classification (default: openai).
        reliability_threshold: Minimum reliability score for taxonomy mode.
        legacy_classify: Force legacy GPT binary classification instead of taxonomy.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    use_taxonomy = (config.CLASSIFICATION_MODE == "taxonomy") and not legacy_classify

    # Step 1: Classification
    if use_taxonomy:
        if skip_classify and os.path.exists(config.FULL_CLASSIFICATION_CSV):
            print(f"Skipping classification (using existing {config.FULL_CLASSIFICATION_CSV})")
        else:
            print("=" * 60)
            print("STEP 1: Taxonomy-based multi-model classification")
            print("=" * 60)
            run_full_classification(
                num_samples=num_samples,
                resume=resume,
                delay=delay,
                providers=providers or ["openai"],
                reliability_threshold=0.66,
            )
        # Generate legacy bridge CSVs for downstream steps
        if os.path.exists(config.FULL_CLASSIFICATION_CSV):
            _generate_legacy_csvs(config.FULL_CLASSIFICATION_CSV)
    else:
        if skip_classify and os.path.exists(config.CLASSIFICATION_CSV):
            print(f"Skipping classification (using existing {config.CLASSIFICATION_CSV})")
        else:
            print("=" * 60)
            print("STEP 1: Classifying images (legacy GPT binary)")
            print("=" * 60)
            run_classification(
                num_samples=num_samples,
                test_ids=test_ids,
                resume=resume,
                delay=delay,
            )

    # Step 2: Descriptions (3 independent proprietary VLMs)
    if skip_describe and os.path.exists(config.DESCRIPTIONS_CSV):
        print(f"Skipping descriptions (using existing {config.DESCRIPTIONS_CSV})")
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Generating descriptions (OpenAI + Gemini + Claude)")
        print("=" * 60)
        run_description(resume=resume, delay=delay)

    # Step 3: Aggregation (Qwen via OpenRouter)
    if skip_aggregate and os.path.exists(config.AGGREGATED_CSV):
        print(f"Skipping aggregation (using existing {config.AGGREGATED_CSV})")
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Qwen aggregation (open-source VLM)")
        print("=" * 60)
        aggregated_df = run_consensus(resume=resume)

    # Step 4: HTML Report
    print("\n" + "=" * 60)
    print("STEP 4: Generating HTML report")
    print("=" * 60)
    report_path = generate_report()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if use_taxonomy:
        print(f"  Full classification: {config.FULL_CLASSIFICATION_CSV}")
    print(f"  Classification: {config.CLASSIFICATION_CSV}")
    print(f"  Skipped:        {config.SKIPPED_CSV}")
    print(f"  Descriptions:   {config.DESCRIPTIONS_CSV}")
    print(f"  Aggregated:     {config.AGGREGATED_CSV}")
    print(f"  Report:         {report_path}")

    return aggregated_df if 'aggregated_df' in dir() else None


def run_benchmarking_pipeline(
    num_samples: int | None = None,
    test_ids: list | None = None,
    resume: bool = True,
    delay: float | None = None,
    skip_prep: bool = False,
    skip_describe: bool = False,
    skip_retry: bool = False,
    skip_synthesize: bool = False,
) -> None:
    """
    Classification-free benchmarking pipeline.

    Steps:
      1. prepare_all_images()   — build all_images.csv (no API calls)
      2. run_description()      — 4 VLM providers in parallel
      3. run_prompt_synthesis() — Llama 3.3-70B judge → concise_prompt
      4. generate_benchmarking_report() — image + concise_prompt HTML

    Args:
        num_samples: Limit to first N images. None = all 3040.
        test_ids: Only process specific image IDs (e.g. [363, 364, 365]).
        resume: Resume each step from existing checkpoints.
        delay: Seconds between API calls.
        skip_prep: Skip step 1 if all_images.csv already exists.
        skip_describe: Skip step 2 if descriptions.csv already exists.
        skip_synthesize: Skip step 3 if concise_prompts.csv already exists.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if skip_prep and os.path.exists(config.ALL_IMAGES_CSV):
        print(f"Skipping prep  (using existing {config.ALL_IMAGES_CSV})")
    else:
        print("=" * 60)
        print("STEP 1: Preparing all images (bypassing classification)")
        print("=" * 60)
        prepare_all_images(num_samples=num_samples, test_ids=test_ids)

    if skip_describe and os.path.exists(config.DESCRIPTIONS_CSV):
        print(f"Skipping describe (using existing {config.DESCRIPTIONS_CSV})")
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Generating descriptions (5 providers in parallel)")
        print("=" * 60)
        run_description(input_csv=config.ALL_IMAGES_CSV, resume=resume, delay=delay)

    if not skip_retry:
        print("\n" + "=" * 60)
        print("STEP 2b: Retrying any failed provider descriptions")
        print("=" * 60)
        retry_failed_descriptions(delay=delay)

    if skip_synthesize and os.path.exists(config.CONCISE_PROMPTS_CSV):
        print(f"Skipping synthesis (using existing {config.CONCISE_PROMPTS_CSV})")
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Synthesizing concise prompts (Llama 3.3-70B judge)")
        print("=" * 60)
        run_prompt_synthesis(resume=resume, delay=delay)

    print("\n" + "=" * 60)
    print("STEP 4: Generating benchmarking HTML report")
    print("=" * 60)
    report_path = generate_benchmarking_report()

    print("\n" + "=" * 60)
    print("BENCHMARKING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  All images CSV:   {config.ALL_IMAGES_CSV}")
    print(f"  Descriptions:     {config.DESCRIPTIONS_CSV}")
    print(f"  Concise prompts:  {config.CONCISE_PROMPTS_CSV}")
    print(f"  Report:           {report_path}")


def run_datikz_pipeline(
    num_samples: int | None = None,
    test_ids: list | None = None,
    resume: bool = True,
    delay: float | None = None,
    providers: list[str] | None = None,
    skip_prep: bool = False,
    skip_describe: bool = False,
    skip_retry: bool = False,
    skip_synthesize: bool = False,
) -> None:
    """
    Benchmarking pipeline for the DaTikZ v3 dataset (nllg/datikz-v3).

    Isolated from the MathVision pipeline — all outputs go to output/datikz/.

    Steps:
      1. prepare_datikz_images()   — build output/datikz/all_images.csv
      2. run_description()         — Gemini + Llama (default) in parallel
      2b. retry_failed_descriptions() — patch any errors in-place
      3. run_prompt_synthesis()    — Llama 3.3-70B judge → concise_prompt
      4. generate_benchmarking_report() — image + prompt HTML

    Args:
        num_samples: Limit to first N images. None = all ~148k.
        test_ids: Only process specific image IDs.
        resume: Resume each step from existing checkpoints.
        delay: Seconds between API calls.
        providers: VLM providers to use. Defaults to ["gemini", "llama"].
        skip_prep: Skip step 1 if all_images.csv already exists.
        skip_describe: Skip step 2 if descriptions.csv already exists.
        skip_retry: Skip step 2b retry pass.
        skip_synthesize: Skip step 3 if concise_prompts.csv already exists.
    """
    providers = providers or ["gemini", "llama"]
    os.makedirs(config.DATIKZ_OUTPUT_DIR, exist_ok=True)

    provider_columns = [(p.capitalize(), f"description_{p}") for p in providers]

    def _datikz_image_loader(image_id):
        pil = get_datikz_image_pil(image_id)
        b64, _ = get_datikz_image_base64(image_id) if pil else (None, None)
        return pil, b64

    if skip_prep and os.path.exists(config.DATIKZ_ALL_IMAGES_CSV):
        print(f"Skipping prep (using existing {config.DATIKZ_ALL_IMAGES_CSV})")
    else:
        print("=" * 60)
        print("DATIKZ STEP 1: Preparing images from nllg/datikz-v3")
        print("=" * 60)
        prepare_datikz_images(num_samples=num_samples, test_ids=test_ids)

    if skip_describe and os.path.exists(config.DATIKZ_DESCRIPTIONS_CSV):
        print(f"Skipping describe (using existing {config.DATIKZ_DESCRIPTIONS_CSV})")
    else:
        print("\n" + "=" * 60)
        print(f"DATIKZ STEP 2: Generating descriptions ({', '.join(providers)})")
        print("=" * 60)
        run_description(
            input_csv=config.DATIKZ_ALL_IMAGES_CSV,
            output_csv=config.DATIKZ_DESCRIPTIONS_CSV,
            resume=resume,
            delay=delay,
            providers=providers,
            image_loader=_datikz_image_loader,
        )

    if not skip_retry:
        print("\n" + "=" * 60)
        print("DATIKZ STEP 2b: Retrying any failed provider descriptions")
        print("=" * 60)
        retry_failed_descriptions(delay=delay, descriptions_csv=config.DATIKZ_DESCRIPTIONS_CSV, providers=providers)

    if skip_synthesize and os.path.exists(config.DATIKZ_CONCISE_PROMPTS_CSV):
        print(f"Skipping synthesis (using existing {config.DATIKZ_CONCISE_PROMPTS_CSV})")
    else:
        print("\n" + "=" * 60)
        print("DATIKZ STEP 3: Synthesizing concise prompts (Llama 3.3-70B judge)")
        print("=" * 60)
        run_prompt_synthesis(
            input_csv=config.DATIKZ_DESCRIPTIONS_CSV,
            output_csv=config.DATIKZ_CONCISE_PROMPTS_CSV,
            resume=resume,
            delay=delay,
            provider_columns=provider_columns,
        )

    print("\n" + "=" * 60)
    print("DATIKZ STEP 4: Generating HTML report")
    print("=" * 60)
    report_path = generate_benchmarking_report(
        input_csv=config.DATIKZ_CONCISE_PROMPTS_CSV,
        output_path=config.DATIKZ_REPORT,
    )

    print("\n" + "=" * 60)
    print("DATIKZ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  All images CSV:   {config.DATIKZ_ALL_IMAGES_CSV}")
    print(f"  Descriptions:     {config.DATIKZ_DESCRIPTIONS_CSV}")
    print(f"  Concise prompts:  {config.DATIKZ_CONCISE_PROMPTS_CSV}")
    print(f"  Report:           {report_path}")


def main():
    parser = argparse.ArgumentParser(description="MathVDiagram benchmarking pipeline")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit to first N samples")
    parser.add_argument("--test-ids", type=int, nargs="+", default=None, help="Specific image IDs to process")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume from checkpoints")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between API calls")
    parser.add_argument("--skip-classify", action="store_true", help="Skip classification if CSV exists")
    parser.add_argument("--skip-describe", action="store_true", help="Skip descriptions if CSV exists")
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip aggregation if CSV exists")
    parser.add_argument(
        "--providers", nargs="+", default=["openai"],
        help="LLM providers for taxonomy classification (default: openai)",
    )
    parser.add_argument(
        "--reliability-threshold", type=float, default=0.7,
        help="Reliability score threshold for taxonomy mode (default: 0.7)",
    )
    parser.add_argument(
        "--legacy-classify", action="store_true",
        help="Use legacy GPT binary classification instead of taxonomy",
    )
    args = parser.parse_args()

    run_pipeline(
        num_samples=args.num_samples,
        test_ids=args.test_ids,
        resume=not args.no_resume,
        delay=args.delay,
        skip_classify=args.skip_classify,
        skip_describe=args.skip_describe,
        skip_aggregate=args.skip_aggregate,
        providers=args.providers,
        reliability_threshold=args.reliability_threshold,
        legacy_classify=args.legacy_classify,
    )


if __name__ == "__main__":
    main()
