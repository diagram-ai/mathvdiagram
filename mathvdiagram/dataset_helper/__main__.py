"""
CLI entry point for the dataset_helper subpackage.

Usage:
    python -m mathvdiagram.dataset_helper --classify --num-samples 100
    python -m mathvdiagram.dataset_helper --classify --providers openai gemini --report
    python -m mathvdiagram.dataset_helper --report
    python -m mathvdiagram.dataset_helper --stats
    python -m mathvdiagram.dataset_helper --export-taxonomy
    python -m mathvdiagram.dataset_helper --validate
    python -m mathvdiagram.dataset_helper --ablation
    python -m mathvdiagram.dataset_helper --paper-stats
"""

from __future__ import annotations

import argparse

from ..data_loader import load_mathvision
from .exploration import print_dataset_summary
from .pipeline import (
    run_full_classification,
    generate_paper_statistics,
    export_taxonomy_document,
)
from .report import generate_classification_report
from .validation import (
    validate_classifications_with_descriptions,
    compute_validation_statistics,
    run_filter_ablation,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mathvdiagram.dataset_helper",
        description="MathVision dataset helper — taxonomy-aware classification and validation.",
    )

    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--classify", action="store_true", help="Run full classification pipeline")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit to N samples")
    parser.add_argument(
        "--providers", nargs="+", default=["openai"],
        help="LLM providers for classification (default: openai). Use multiple for consensus: --providers openai gemini",
    )
    parser.add_argument("--reliability-threshold", type=float, default=0.7,
                        help="Reliability score threshold (default: 0.7)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoints")
    parser.add_argument("--delay", type=float, default=None, help="Delay between API requests (seconds)")
    parser.add_argument("--validate", action="store_true",
                        help="Run downstream quality validation (requires descriptions.csv)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run filter ablation study")
    parser.add_argument("--paper-stats", action="store_true",
                        help="Generate paper-ready statistics")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML classification report")
    parser.add_argument("--export-taxonomy", action="store_true",
                        help="Export taxonomy document")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path override")

    args = parser.parse_args(argv)

    ran_something = False

    if args.stats:
        df = load_mathvision()
        print_dataset_summary(df)
        ran_something = True

    if args.classify:
        run_full_classification(
            num_samples=args.num_samples,
            resume=not args.no_resume,
            delay=args.delay,
            providers=args.providers,
            reliability_threshold=args.reliability_threshold,
            seed=args.seed,
        )
        ran_something = True

    if args.report:
        generate_classification_report(output_path=args.output)
        ran_something = True

    if args.validate:
        validated = validate_classifications_with_descriptions()
        compute_validation_statistics(validated)
        ran_something = True

    if args.ablation:
        run_filter_ablation(seed=args.seed)
        ran_something = True

    if args.paper_stats:
        generate_paper_statistics(output_path=args.output)
        ran_something = True

    if args.export_taxonomy:
        export_taxonomy_document(output_path=args.output)
        ran_something = True

    if not ran_something:
        parser.print_help()


if __name__ == "__main__":
    main()
