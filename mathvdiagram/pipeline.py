"""
Main pipeline orchestrator.

Runs the full benchmarking pipeline:
  Step 1: Classify images as math/non-math (GPT-4o-mini)
  Step 2: Generate descriptions from Gemini + OpenAI for math images
  Step 3: Claude consensus engine to produce final detailed + concise prompts
"""

import os
import argparse

from . import config
from .classify import run_classification
from .describe import run_description
from .consensus import run_consensus
from .report import generate_report


def run_pipeline(
    num_samples: int | None = None,
    test_ids: list | None = None,
    resume: bool = True,
    delay: float | None = None,
    skip_classify: bool = False,
    skip_describe: bool = False,
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
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Step 1: Classification
    if skip_classify and os.path.exists(config.CLASSIFICATION_CSV):
        print(f"Skipping classification (using existing {config.CLASSIFICATION_CSV})")
    else:
        print("=" * 60)
        print("STEP 1: Classifying images (math vs non-math)")
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
    args = parser.parse_args()

    run_pipeline(
        num_samples=args.num_samples,
        test_ids=args.test_ids,
        resume=not args.no_resume,
        delay=args.delay,
        skip_classify=args.skip_classify,
        skip_describe=args.skip_describe,
    )


if __name__ == "__main__":
    main()
