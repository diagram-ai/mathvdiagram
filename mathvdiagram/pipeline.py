"""
Main pipeline orchestrator.

Runs the full benchmarking pipeline:
  Step 1: Classify images — taxonomy-based multi-model (default) or legacy GPT binary
  Step 2: Generate descriptions from Gemini + OpenAI for math/diagram images
  Step 3: Claude consensus engine to produce final detailed + concise prompts
"""

import os
import argparse

import pandas as pd

from . import config
from .classify import run_classification
from .describe import run_description
from .consensus import run_consensus
from .report import generate_report
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
    providers: list[str] | None = None,
    reliability_threshold: float = 0.7,
    legacy_classify: bool = False,
):
    """
    Run the full pipeline from classification through consensus.

    Args:
        num_samples: Limit to first N samples. None = all.
        test_ids: Only process specific IDs. None = all.
        resume: Resume from existing checkpoints.
        delay: Seconds between API calls.
        skip_classify: Skip step 1 if classification CSV already exists.
        skip_describe: Skip step 2 if descriptions CSV already exists.
        providers: LLM providers for taxonomy classification (default: openai, gemini, claude).
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

    # Step 2: Descriptions
    if skip_describe and os.path.exists(config.DESCRIPTIONS_CSV):
        print(f"Skipping descriptions (using existing {config.DESCRIPTIONS_CSV})")
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Generating descriptions (Gemini + OpenAI)")
        print("=" * 60)
        run_description(resume=resume, delay=delay)

    # Step 3: Consensus
    print("\n" + "=" * 60)
    print("STEP 3: Claude consensus engine")
    print("=" * 60)
    consensus_df = run_consensus(resume=resume)

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
    print(f"  Consensus:      {config.CONSENSUS_CSV}")
    print(f"  Report:         {report_path}")

    return consensus_df


def main():
    parser = argparse.ArgumentParser(description="MathVDiagram benchmarking pipeline")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit to first N samples")
    parser.add_argument("--test-ids", type=int, nargs="+", default=None, help="Specific image IDs to process")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume from checkpoints")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between API calls")
    parser.add_argument("--skip-classify", action="store_true", help="Skip classification if CSV exists")
    parser.add_argument("--skip-describe", action="store_true", help="Skip descriptions if CSV exists")
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
        providers=args.providers,
        reliability_threshold=args.reliability_threshold,
        legacy_classify=args.legacy_classify,
    )


if __name__ == "__main__":
    main()
