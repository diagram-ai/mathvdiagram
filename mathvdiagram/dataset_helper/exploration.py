"""
Dataset exploration, statistics, sampling, and metadata pre-filtering.
"""

from __future__ import annotations

import os

import pandas as pd

from .. import config
from .taxonomy import SUBJECT_PRIORITY


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the MathVision dataset."""
    subject_dist = df["subject"].value_counts().to_dict()
    level_dist = df["level"].value_counts().sort_index().to_dict()

    cross = df.groupby(["subject", "level"]).size().unstack(fill_value=0)
    subject_by_level = {subj: row.to_dict() for subj, row in cross.iterrows()}

    priority_dist: dict[str, int] = {"high": 0, "mixed": 0, "low": 0, "unknown": 0}
    for subj, count in subject_dist.items():
        tier = SUBJECT_PRIORITY.get(subj, "unknown")
        priority_dist[tier] += count

    return {
        "total_images": len(df),
        "subject_distribution": subject_dist,
        "level_distribution": level_dist,
        "subject_by_level": subject_by_level,
        "priority_distribution": priority_dist,
    }


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Pretty-print dataset statistics to console."""
    stats = get_dataset_statistics(df)

    print("=" * 70)
    print(f"MATHVISION DATASET SUMMARY  ({stats['total_images']} images)")
    print("=" * 70)

    print("\nSubject Distribution:")
    print(f"  {'Subject':<35} {'Count':>6}  {'Priority':>8}")
    print("  " + "-" * 53)
    for subj, count in sorted(stats["subject_distribution"].items(), key=lambda x: -x[1]):
        tier = SUBJECT_PRIORITY.get(subj, "?")
        print(f"  {subj:<35} {count:>6}  {tier:>8}")

    print(f"\nDifficulty Level Distribution:")
    for level, count in sorted(stats["level_distribution"].items()):
        print(f"  Level {level}: {count}")

    print(f"\nPriority Tier Distribution:")
    for tier in ["high", "mixed", "low", "unknown"]:
        count = stats["priority_distribution"].get(tier, 0)
        if count:
            print(f"  {tier:<8}: {count}")

    print("=" * 70)


def sample_by_subject(df: pd.DataFrame, n_per_subject: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return a stratified sample with n images per subject."""
    return (
        df.groupby("subject", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_subject, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def get_annotation_sample(
    df: pd.DataFrame,
    n_total: int = 300,
    stratify_by: str = "subject",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a proportionally stratified sample for human annotation.
    Saves sample IDs to CSV for reproducibility.
    """
    groups = df.groupby(stratify_by)
    total = len(df)
    samples = []
    for _, group in groups:
        proportion = len(group) / total
        n = max(1, round(proportion * n_total))
        n = min(n, len(group))
        samples.append(group.sample(n=n, random_state=seed))

    result = pd.concat(samples).reset_index(drop=True)
    result = result[["id", "question", "subject", "level"]]

    out_path = os.path.join(config.OUTPUT_DIR, "annotation_sample_ids.csv")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Annotation sample ({len(result)} images) saved to {out_path}")
    return result


def apply_metadata_prefilter(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Layer 1: split dataset by subject priority tier (no API calls).

    Returns dict:
      - "likely_diagram": high-priority subjects
      - "needs_inspection": mixed-priority subjects
      - "likely_non_diagram": low-priority subjects
    """
    tier_map = {"high": "likely_diagram", "mixed": "needs_inspection", "low": "likely_non_diagram"}
    tiers: dict[str, list] = {"likely_diagram": [], "needs_inspection": [], "likely_non_diagram": []}

    for _, row in df.iterrows():
        bucket = tier_map.get(SUBJECT_PRIORITY.get(row["subject"], "mixed"), "needs_inspection")
        tiers[bucket].append(row)

    result = {k: pd.DataFrame(v) for k, v in tiers.items()}

    print(f"\nMetadata pre-filter results:")
    print(f"  Include via metadata (high tier): {len(result['likely_diagram'])} images — no API calls needed")
    print(f"  Needs LLM inspection (mixed tier): {len(result['needs_inspection'])} images")
    print(f"  Likely non-diagram (low tier): {len(result['likely_non_diagram'])} images")
    print(f"  Total API calls needed: {len(result['needs_inspection']) + len(result['likely_non_diagram'])} (saved {len(result['likely_diagram'])} calls)")

    return result
