"""
Downstream quality signals and self-validation (Layer 4).

Validates classification results using description quality signals
and supports ablation studies to prove the filter adds value.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from .. import config


# Keywords that indicate an image is NOT a proper diagram
NON_DIAGRAM_SIGNAL_PHRASES: list[str] = [
    "photograph", "real-world", "real world", "illustration of", "cartoon",
    "clipart", "picture of", "shows a photo", "decorative", "artistic",
    "depicts an animal", "depicts a person", "toy", "food", "flower",
]

NEGATION_PREFIXES: list[str] = [
    "not a ", "not an ", "no ", "isn't a ", "isn't an ", "is not a ", "is not an ",
    "without ", "unlike ", "rather than ", "instead of ", "absence of ",
    "don't ", "doesn't ", "never ",
]


def _is_negated(description_lower: str, phrase: str, match_pos: int) -> bool:
    """Check if a phrase match is preceded by a negation within a short window."""
    window_start = max(0, match_pos - 30)
    prefix_window = description_lower[window_start:match_pos]
    return any(neg in prefix_window for neg in NEGATION_PREFIXES)


# Keywords that indicate mathematical content
MATH_ELEMENT_PHRASES: list[str] = [
    "axis", "axes", "vertex", "vertices", "angle", "coordinate",
    "triangle", "circle", "polygon", "graph", "plot", "function",
    "equation", "labeled", "perpendicular", "parallel", "dimension",
]


def detect_quality_signals(description: str) -> dict:
    """
    Analyze a generated description for signals that suggest misclassification.

    Returns dict with: description_length, is_too_short, non_diagram_phrases_found,
    has_non_diagram_signals, mentions_mathematical_elements, quality_flag.
    """
    if not description or not isinstance(description, str):
        return {
            "description_length": 0,
            "is_too_short": True,
            "non_diagram_phrases_found": [],
            "has_non_diagram_signals": False,
            "mentions_mathematical_elements": False,
            "quality_flag": "suspect",
        }

    desc_lower = description.lower()
    length = len(description)
    is_too_short = length < 100

    # Negation-aware phrase matching (FIX 5)
    non_diag_found = []
    for phrase in NON_DIAGRAM_SIGNAL_PHRASES:
        for match in re.finditer(re.escape(phrase), desc_lower):
            if not _is_negated(desc_lower, phrase, match.start()):
                non_diag_found.append(phrase)
                break  # one confirmed match per phrase is enough
    has_non_diag = len(non_diag_found) > 0

    math_found = False
    for phrase in MATH_ELEMENT_PHRASES:
        for match in re.finditer(re.escape(phrase), desc_lower):
            if not _is_negated(desc_lower, phrase, match.start()):
                math_found = True
                break
        if math_found:
            break
    has_math = math_found

    # Determine quality flag
    if has_non_diag and not has_math:
        flag = "likely_misclassified"
    elif has_non_diag or is_too_short:
        flag = "suspect"
    else:
        flag = "good"

    return {
        "description_length": length,
        "is_too_short": is_too_short,
        "non_diagram_phrases_found": non_diag_found,
        "has_non_diagram_signals": has_non_diag,
        "mentions_mathematical_elements": has_math,
        "quality_flag": flag,
    }


def validate_classifications_with_descriptions(
    classification_csv: str | None = None,
    descriptions_csv: str | None = None,
) -> pd.DataFrame:
    """
    Cross-reference classifications with generated descriptions.

    For each image classified as a diagram (is_diagram=True), check description
    quality using signal detection.

    Returns DataFrame with quality signal columns appended.
    """
    classification_csv = classification_csv or os.path.join(config.OUTPUT_DIR, "full_classification.csv")
    descriptions_csv = descriptions_csv or os.path.join(config.OUTPUT_DIR, "descriptions.csv")

    cls_df = pd.read_csv(classification_csv)
    desc_df = pd.read_csv(descriptions_csv)

    # Normalize ID columns for merge
    for df in (cls_df, desc_df):
        if "image_id" in df.columns:
            df["image_id"] = df["image_id"].astype(str)

    # Filter to diagrams only
    if "is_diagram" in cls_df.columns:
        diagrams = cls_df[cls_df["is_diagram"] == True].copy()
    else:
        diagrams = cls_df.copy()

    # Merge with descriptions
    desc_col = None
    for col in ["gemini_description", "openai_description", "description", "consensus_detailed"]:
        if col in desc_df.columns:
            desc_col = col
            break

    if desc_col is None:
        print("No description column found in descriptions CSV.")
        return diagrams

    merged = diagrams.merge(
        desc_df[["image_id", desc_col]],
        on="image_id",
        how="left",
    )

    # Run quality signal detection
    signals = merged[desc_col].fillna("").apply(detect_quality_signals)
    signal_df = pd.DataFrame(signals.tolist())

    result = pd.concat([merged, signal_df], axis=1)

    # Print summary
    total = len(result)
    good = (result["quality_flag"] == "good").sum()
    suspect = (result["quality_flag"] == "suspect").sum()
    misclassified = (result["quality_flag"] == "likely_misclassified").sum()

    print(f"\nQuality validation of {total} classified diagrams:")
    print(f"  Good:                   {good} ({good/total:.1%})" if total else "")
    print(f"  Suspect:                {suspect} ({suspect/total:.1%})" if total else "")
    print(f"  Likely misclassified:   {misclassified} ({misclassified/total:.1%})" if total else "")

    return result


def compute_validation_statistics(validated_df: pd.DataFrame) -> dict:
    """
    Compute statistics for the paper's validation section.

    Returns dict with totals, percentages, and breakdowns by subject/category.
    """
    total = len(validated_df)
    if total == 0:
        return {"total_diagrams": 0}

    good = int((validated_df["quality_flag"] == "good").sum())
    suspect = int((validated_df["quality_flag"] == "suspect").sum())
    misclassified = int((validated_df["quality_flag"] == "likely_misclassified").sum())

    stats: dict = {
        "total_diagrams": total,
        "good_count": good,
        "good_pct": round(good / total, 4),
        "suspect_count": suspect,
        "suspect_pct": round(suspect / total, 4),
        "misclassified_count": misclassified,
        "misclassified_pct": round(misclassified / total, 4),
    }

    # Breakdown by subject
    if "subject" in validated_df.columns:
        flags_by_subject = {}
        for subj, group in validated_df.groupby("subject"):
            n = len(group)
            flags_by_subject[subj] = {
                "total": n,
                "good": int((group["quality_flag"] == "good").sum()),
                "suspect": int((group["quality_flag"] == "suspect").sum()),
                "misclassified": int((group["quality_flag"] == "likely_misclassified").sum()),
            }
        stats["flags_by_subject"] = flags_by_subject

    # Breakdown by category
    cat_col = "majority_category" if "majority_category" in validated_df.columns else "final_category"
    if cat_col in validated_df.columns:
        flags_by_category = {}
        for cat, group in validated_df.groupby(cat_col):
            n = len(group)
            flags_by_category[cat] = {
                "total": n,
                "good": int((group["quality_flag"] == "good").sum()),
                "suspect": int((group["quality_flag"] == "suspect").sum()),
                "misclassified": int((group["quality_flag"] == "likely_misclassified").sum()),
            }
        stats["flags_by_category"] = flags_by_category

    # Print formatted summary
    print("=" * 60)
    print("VALIDATION STATISTICS")
    print("=" * 60)
    print(f"  Total diagrams:         {total}")
    print(f"  Good quality:           {good} ({stats['good_pct']:.1%})")
    print(f"  Suspect:                {suspect} ({stats['suspect_pct']:.1%})")
    print(f"  Likely misclassified:   {misclassified} ({stats['misclassified_pct']:.1%})")

    if "flags_by_category" in stats:
        print("\n  By category:")
        for cat, d in sorted(stats["flags_by_category"].items()):
            print(f"    {cat:<30} good={d['good']}  suspect={d['suspect']}  misclass={d['misclassified']}")

    print("=" * 60)
    return stats


def compute_independent_metrics(descriptions: list[str]) -> dict:
    """
    Compute quality metrics that are independent of the classification criteria.

    Measures description length, math term density, and specificity (labels,
    numbers, units) to provide ablation evidence that doesn't circularly
    depend on the filtering criteria.
    """
    MATH_TERMS = [
        "angle", "triangle", "circle", "vertex", "vertices", "perpendicular",
        "parallel", "coordinate", "x-axis", "y-axis", "origin", "radius",
        "diameter", "hypotenuse", "tangent", "parabola", "slope", "intercept",
        "vector", "matrix", "equation", "inequality", "function", "graph",
        "polygon", "quadrilateral", "pentagon", "hexagon", "arc", "bisector",
        "congruent", "similar", "theorem", "proof", "degree", "radian",
    ]

    lengths = []
    math_term_counts = []
    specificity_scores = []

    for desc in descriptions:
        if not desc or not isinstance(desc, str):
            lengths.append(0)
            math_term_counts.append(0)
            specificity_scores.append(0)
            continue

        desc_lower = desc.lower()
        lengths.append(len(desc))

        # Count unique math terms
        found_terms = set()
        for term in MATH_TERMS:
            if term in desc_lower:
                found_terms.add(term)
        math_term_counts.append(len(found_terms))

        # Specificity: labels (A, B, C...), numbers, measurement units
        labels = len(re.findall(r'\b[A-Z]\b', desc))
        numbers = len(re.findall(r'\b\d+\.?\d*\b', desc))
        units = len(re.findall(r'\b(cm|mm|m|degrees?|°|radians?)\b', desc_lower))
        specificity_scores.append(labels + numbers + units)

    return {
        "mean_description_length": round(np.mean(lengths), 1) if lengths else 0,
        "mean_math_term_count": round(np.mean(math_term_counts), 2) if math_term_counts else 0,
        "mean_specificity_score": round(np.mean(specificity_scores), 2) if specificity_scores else 0,
        "median_description_length": round(float(np.median(lengths)), 1) if lengths else 0,
        "pct_short_descriptions": round(sum(1 for l in lengths if l < 100) / max(len(lengths), 1), 4),
    }


def run_filter_ablation(
    n_sample: int = 50,
    seed: int = 42,
) -> dict:
    """
    Ablation study: does the filter actually matter?

    1. Sample n_sample images that were EXCLUDED (non_diagram or unreliable)
    2. Generate descriptions for them
    3. Compare quality signals between included vs excluded images

    NOTE: This function is expensive (API calls). Results are saved to
    OUTPUT_DIR/ablation_results.json.
    """
    from ..describe import describe_single_image

    classification_csv = os.path.join(config.OUTPUT_DIR, "full_classification.csv")
    if not os.path.exists(classification_csv):
        print("No classification results found. Run --classify first.")
        return {}

    cls_df = pd.read_csv(classification_csv)

    # Get excluded images (non-diagram or unreliable)
    if "is_diagram" in cls_df.columns:
        excluded = cls_df[cls_df["is_diagram"] == False]
        included = cls_df[cls_df["is_diagram"] == True]
    else:
        print("No is_diagram column. Cannot run ablation.")
        return {}

    n_exc = min(n_sample, len(excluded))
    n_inc = min(n_sample, len(included))

    exc_sample = excluded.sample(n=n_exc, random_state=seed)
    inc_sample = included.sample(n=n_inc, random_state=seed)

    print(f"Ablation: generating descriptions for {n_exc} excluded + {n_inc} included images...")

    # Check if descriptions already exist
    descriptions_csv = os.path.join(config.OUTPUT_DIR, "descriptions.csv")
    existing_desc = {}
    if os.path.exists(descriptions_csv):
        desc_df = pd.read_csv(descriptions_csv)
        for _, row in desc_df.iterrows():
            for col in ["gemini_description", "openai_description", "description"]:
                if col in desc_df.columns and pd.notna(row.get(col)):
                    existing_desc[str(row.get("image_id", ""))] = str(row[col])
                    break

    def _get_signals_and_descriptions(sample_df: pd.DataFrame) -> tuple[list[dict], list[str]]:
        signals = []
        descriptions = []
        for _, row in sample_df.iterrows():
            img_id = str(row.get("image_id", row.get("id", "")))
            desc = existing_desc.get(img_id, "")
            descriptions.append(desc)
            signals.append(detect_quality_signals(desc))
        return signals, descriptions

    exc_signals, exc_descriptions = _get_signals_and_descriptions(exc_sample)
    inc_signals, inc_descriptions = _get_signals_and_descriptions(inc_sample)

    def _flag_dist(signals: list[dict]) -> dict:
        n = len(signals)
        if n == 0:
            return {"good": 0, "suspect": 0, "likely_misclassified": 0}
        flags = [s["quality_flag"] for s in signals]
        return {
            "good": flags.count("good"),
            "suspect": flags.count("suspect"),
            "likely_misclassified": flags.count("likely_misclassified"),
            "good_rate": round(flags.count("good") / n, 4),
            "suspect_rate": round(flags.count("suspect") / n, 4),
            "misclass_rate": round(flags.count("likely_misclassified") / n, 4),
        }

    exc_dist = _flag_dist(exc_signals)
    inc_dist = _flag_dist(inc_signals)

    # FIX 7: Independent metrics that don't circularly depend on filter criteria
    exc_metrics = compute_independent_metrics(exc_descriptions)
    inc_metrics = compute_independent_metrics(inc_descriptions)

    result = {
        "excluded_sample_size": n_exc,
        "included_sample_size": n_inc,
        "excluded_sample_quality": {
            "signal_based": exc_dist,
            "independent_metrics": exc_metrics,
        },
        "included_sample_quality": {
            "signal_based": inc_dist,
            "independent_metrics": inc_metrics,
        },
        "quality_difference": {
            "signal_based": {
                "good_rate_diff": round(inc_dist.get("good_rate", 0) - exc_dist.get("good_rate", 0), 4),
                "note": "Positive diff means included images have higher good rate (filter works).",
            },
            "independent_metrics": {
                "length_difference": round(inc_metrics["mean_description_length"] - exc_metrics["mean_description_length"], 1),
                "math_term_difference": round(inc_metrics["mean_math_term_count"] - exc_metrics["mean_math_term_count"], 2),
                "specificity_difference": round(inc_metrics["mean_specificity_score"] - exc_metrics["mean_specificity_score"], 2),
            },
        },
    }

    output_path = os.path.join(config.OUTPUT_DIR, "ablation_results.json")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Ablation results saved to {output_path}")

    print(f"\nAblation results (signal-based):")
    print(f"  Included images — good: {inc_dist.get('good_rate', 0):.1%}")
    print(f"  Excluded images — good: {exc_dist.get('good_rate', 0):.1%}")
    print(f"  Difference: {result['quality_difference']['signal_based']['good_rate_diff']:+.1%}")

    print(f"\n  Independent metrics comparison:")
    print(f"    {'Metric':<30} {'Included':>12} {'Excluded':>12} {'Diff':>10}")
    print(f"    {'-'*64}")
    for key in ["mean_description_length", "mean_math_term_count", "mean_specificity_score"]:
        inc_val = inc_metrics[key]
        exc_val = exc_metrics[key]
        diff = inc_val - exc_val
        print(f"    {key:<30} {inc_val:>12} {exc_val:>12} {diff:>+10.1f}")

    return result
