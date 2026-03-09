"""
Multi-model agreement and reliability scoring (Layer 3).
"""

from __future__ import annotations

from collections import Counter
import json

import numpy as np
import pandas as pd


CONFIDENCE_WEIGHTS: dict[str, float] = {"high": 1.0, "medium": 0.66, "low": 0.33}


def compute_reliability_score(row: pd.Series, providers: list[str]) -> float:
    """
    Compute reliability score for a single image.

    reliability = (agreement_count / num_models) * mean_confidence_of_agreeing_models

    Only confidence weights from models that voted for the majority category
    are included in the mean.  This prevents a disagreeing model's confidence
    from skewing the score.

    Score ranges from ~0.11 (all disagree, all low) to 1.0 (all agree, all high).
    """
    cat_conf_pairs = []
    for prov in providers:
        cat = row.get(f"cat_{prov}")
        conf = row.get(f"conf_{prov}")
        if pd.notna(cat) and cat != "unknown":
            w = CONFIDENCE_WEIGHTS.get(str(conf), 0.33) if pd.notna(conf) else 0.33
            cat_conf_pairs.append((cat, w))

    if not cat_conf_pairs:
        return 0.0

    categories = [c for c, _ in cat_conf_pairs]
    counter = Counter(categories)
    majority_cat, agreement_count = counter.most_common(1)[0]
    n_models = len(categories)

    agreeing_confs = [w for c, w in cat_conf_pairs if c == majority_cat]
    mean_agreeing_conf = np.mean(agreeing_confs) if agreeing_confs else 0.33

    return (agreement_count / n_models) * mean_agreeing_conf


# Expected boolean patterns per category — used for consistency checking (FIX 3)
CATEGORY_BOOLEAN_RULES: dict[str, dict[str, list[str]]] = {
    "geometric_construction": {
        "expected_true": ["has_geometric_labels"],
        "expected_false": ["has_real_world_objects", "has_photographic_content"],
    },
    "coordinate_plot": {
        "expected_true": ["has_labeled_axes", "has_grid_or_coordinate_system"],
        "expected_false": ["has_real_world_objects", "has_photographic_content"],
    },
    "statistical_chart": {
        "expected_true": ["has_labeled_axes"],
        "expected_false": ["has_real_world_objects", "has_photographic_content"],
    },
    "schematic_diagram": {
        "expected_true": ["has_mathematical_notation"],
        "expected_false": ["has_real_world_objects", "has_photographic_content"],
    },
    "3d_figure": {
        "expected_true": ["has_geometric_labels"],
        "expected_false": ["has_real_world_objects", "has_photographic_content"],
    },
    "non_diagram": {
        "expected_true": ["has_real_world_objects"],
        "expected_false": [],
    },
}


def check_boolean_category_consistency(classification: dict) -> tuple[bool, int]:
    """
    Check if a classification's boolean features are consistent with its category.

    Returns:
        (is_consistent, violation_count)

    A classification is inconsistent if an ``expected_true`` boolean is False
    or an ``expected_false`` boolean is True.  Rules are soft — violations
    reduce reliability but never hard-reject.
    """
    category = classification.get("diagram_category", "unknown")
    rules = CATEGORY_BOOLEAN_RULES.get(category)
    if rules is None:
        return True, 0

    violations = 0
    for key in rules.get("expected_true", []):
        if classification.get(key) is False:  # explicitly False, not missing
            violations += 1
    for key in rules.get("expected_false", []):
        if classification.get(key) is True:
            violations += 1

    return violations == 0, violations


def classify_from_booleans(row: pd.Series, providers: list[str]) -> tuple[str, str]:
    """
    Make a deterministic diagram/non-diagram decision from boolean features.

    Parses the full cls_{provider} JSON for each provider to extract booleans,
    then applies majority vote on each boolean across providers.
    After determining the consensus booleans, applies deterministic rules.

    Returns:
        (decision: str, reason: str)
        decision is one of: "diagram", "non_diagram", "uncertain", "unclassified"
        reason is a short explanation of which rule triggered
    """
    # Collect boolean votes across providers
    boolean_keys = [
        "has_labeled_axes", "has_geometric_labels", "has_real_world_objects",
        "has_photographic_content", "has_mathematical_notation",
        "has_grid_or_coordinate_system",
    ]

    # For each boolean, collect True/False votes from all providers
    boolean_votes: dict[str, list[bool]] = {k: [] for k in boolean_keys}

    for prov in providers:
        cls_str = row.get(f"cls_{prov}")
        if pd.isna(cls_str):
            continue
        try:
            cls = json.loads(cls_str) if isinstance(cls_str, str) else cls_str
        except (json.JSONDecodeError, TypeError):
            continue

        for key in boolean_keys:
            val = cls.get(key)
            if isinstance(val, bool):
                boolean_votes[key].append(val)

    # If no provider returned valid boolean data, this image was not classified
    total_votes = sum(len(v) for v in boolean_votes.values())
    if total_votes == 0:
        return "unclassified", "no provider data available (API error)"

    # Resolve each boolean by majority vote (default False if no votes)
    bools = {}
    for key in boolean_keys:
        votes = boolean_votes[key]
        if not votes:
            bools[key] = False
        else:
            bools[key] = sum(votes) > len(votes) / 2

    # --- DETERMINISTIC RULES ---

    # Rule 1: Real-world objects WITHOUT any mathematical structure → non_diagram
    has_math_structure = (
        bools["has_geometric_labels"]
        or bools["has_labeled_axes"]
        or bools["has_grid_or_coordinate_system"]
    )
    if bools["has_real_world_objects"] and not has_math_structure:
        return "non_diagram", "real_world_objects without mathematical structure"

    # Rule 2: Photographic content WITHOUT any mathematical structure → non_diagram
    if bools["has_photographic_content"] and not has_math_structure:
        return "non_diagram", "photographic content without mathematical structure"

    # Rule 3: Any strong mathematical structure signal → diagram
    if bools["has_geometric_labels"]:
        return "diagram", "has geometric labels (vertices, angles, congruence marks)"

    if bools["has_labeled_axes"]:
        return "diagram", "has labeled axes"

    if bools["has_grid_or_coordinate_system"]:
        return "diagram", "has grid or coordinate system"

    # Rule 4: Mathematical notation without real-world objects → diagram
    if bools["has_mathematical_notation"] and not bools["has_real_world_objects"]:
        return "diagram", "has mathematical notation without real-world objects"

    # Rule 5: No strong signals either way → uncertain, fall back to model's category
    return "uncertain", "no strong boolean signals"


def compute_agreement(
    classifications: pd.DataFrame,
    providers: list[str] | None = None,
    # 0.66: natural cutoff where unanimous medium-confidence (1.0 * 0.66)
    # just passes, and 2/3 majority with high confidence (0.67 * 1.0) also passes.
    reliability_threshold: float = 0.66,
) -> pd.DataFrame:
    """
    Compute agreement metrics across provider classification columns.

    Adds columns: majority_category, agreement_count, agreement_level,
    mean_confidence, reliability_score, is_reliable, is_diagram, boolean_violations.

    Agreement levels: "unanimous", "majority", "no_agreement", "insufficient_data",
    "single_model" (when only one provider is used).

    In single-provider mode, reliability = the model's confidence weight directly
    (high=1.0, medium=0.66, low=0.33). No boolean consistency penalty is applied.

    In multi-provider mode, requires at least 2 valid (non-unknown) responses.
    Uses only agreeing models' confidence for the reliability score and applies
    a mild penalty for boolean-category inconsistencies.
    """
    if providers is None:
        providers = [
            c.replace("cat_", "")
            for c in classifications.columns
            if c.startswith("cat_")
        ]
    if not providers:
        print("No cat_* columns found (expected columns like cat_openai, cat_gemini, cat_claude)")
        return classifications

    # Single-provider mode: no agreement computation needed
    # The single model's judgment is used directly
    single_provider_mode = len(providers) == 1

    df = classifications.copy()

    majority_cats = []
    agreement_counts = []
    agreement_levels = []
    mean_confs = []
    reliability_scores = []
    boolean_violations_list = []

    # Require at least 2 valid responses for multi-provider, 1 for single
    min_required = 1 if single_provider_mode else min(2, len(providers))

    for _, row in df.iterrows():
        # Single-provider fast path
        if single_provider_mode:
            prov = providers[0]
            cat = row.get(f"cat_{prov}")
            conf = row.get(f"conf_{prov}")

            if pd.isna(cat) or cat == "unknown":
                majority_cats.append("unknown")
                agreement_counts.append(0)
                agreement_levels.append("insufficient_data")
                mean_confs.append(0.0)
                reliability_scores.append(0.0)
                boolean_violations_list.append(0)
            else:
                conf_weight = CONFIDENCE_WEIGHTS.get(str(conf), 0.33) if pd.notna(conf) else 0.33
                majority_cats.append(cat)
                agreement_counts.append(1)
                agreement_levels.append("single_model")
                mean_confs.append(conf_weight)
                reliability_scores.append(conf_weight)  # reliability = confidence
                boolean_violations_list.append(0)
            continue

        # Multi-provider path
        cat_conf_pairs = []
        for prov in providers:
            cat = row.get(f"cat_{prov}")
            conf = row.get(f"conf_{prov}")
            if pd.notna(cat) and cat != "unknown":
                w = CONFIDENCE_WEIGHTS.get(str(conf), 0.33) if pd.notna(conf) else 0.33
                cat_conf_pairs.append((cat, w))

        categories = [c for c, _ in cat_conf_pairs]
        confidences = [w for _, w in cat_conf_pairs]
        n_valid = len(categories)

        if n_valid < min_required:
            majority_cats.append(categories[0] if categories else "unknown")
            agreement_counts.append(n_valid)
            agreement_levels.append("insufficient_data")
            mean_confs.append(round(np.mean(confidences), 4) if confidences else 0.0)
            reliability_scores.append(0.0)
            boolean_violations_list.append(0)
            continue

        counter = Counter(categories)
        most_common = counter.most_common()
        n_models = len(categories)

        # Break ties by highest total confidence weight
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tied_cats = [c for c, cnt in most_common if cnt == most_common[0][1]]
            cat_total_conf = {}
            for cat, w in cat_conf_pairs:
                if cat in tied_cats:
                    cat_total_conf[cat] = cat_total_conf.get(cat, 0.0) + w
            best_cat = max(tied_cats, key=lambda c: cat_total_conf.get(c, 0.0))
            agree_count = counter[best_cat]
        else:
            best_cat = most_common[0][0]
            agree_count = most_common[0][1]

        if agree_count == n_models:
            level = "unanimous"
        elif agree_count > n_models / 2:
            level = "majority"
        else:
            level = "no_agreement"

        agreeing_confs = [w for c, w in cat_conf_pairs if c == best_cat]
        mean_agreeing_conf = np.mean(agreeing_confs) if agreeing_confs else 0.33
        mean_conf = np.mean(confidences) if confidences else 0.0
        rel_score = (agree_count / n_models) * mean_agreeing_conf

        # Boolean consistency penalty (multi-provider only)
        violation_counts = []
        for prov in providers:
            cls_str = row.get(f"cls_{prov}")
            if pd.notna(cls_str):
                try:
                    cls_dict = json.loads(str(cls_str))
                    _, violations = check_boolean_category_consistency(cls_dict)
                    violation_counts.append(violations)
                except (json.JSONDecodeError, TypeError):
                    pass
        max_violations = max(violation_counts) if violation_counts else 0
        penalty = min(max_violations * 0.05, 0.2)  # cap at 0.2 reduction
        rel_score = max(0.0, rel_score - penalty)

        majority_cats.append(best_cat)
        agreement_counts.append(agree_count)
        agreement_levels.append(level)
        mean_confs.append(round(mean_conf, 4))
        reliability_scores.append(round(rel_score, 4))
        boolean_violations_list.append(max_violations)

    df["majority_category"] = majority_cats
    df["agreement_count"] = agreement_counts
    df["agreement_level"] = agreement_levels
    df["mean_confidence"] = mean_confs
    df["reliability_score"] = reliability_scores
    df["boolean_violations"] = boolean_violations_list

    # Boolean-based classification
    boolean_decisions = []
    boolean_reasons = []
    for _, row in df.iterrows():
        decision, reason = classify_from_booleans(row, providers)
        boolean_decisions.append(decision)
        boolean_reasons.append(reason)

    df["boolean_decision"] = boolean_decisions
    df["boolean_reason"] = boolean_reasons

    # Final is_diagram decision:
    # - If booleans say "diagram" → diagram (regardless of model's category or confidence)
    # - If booleans say "non_diagram" or "unclassified" → not a diagram
    # - If booleans say "uncertain" and model category is unknown → not a diagram
    # - If booleans say "uncertain" with a real category → trust model's category
    df["is_diagram"] = df.apply(
        lambda row: (
            True if row["boolean_decision"] == "diagram"
            else False if row["boolean_decision"] in ("non_diagram", "unclassified")
            else False if row["majority_category"] in ("non_diagram", "unknown")
            else True  # uncertain with a real category → trust model
        ),
        axis=1,
    )

    # Keep is_reliable for reporting but it no longer drives the is_diagram decision
    df["is_reliable"] = df["reliability_score"] >= reliability_threshold

    return df


def compute_agreement_statistics(
    df: pd.DataFrame,
    providers: list[str] | None = None,
) -> dict:
    """Compute and print comprehensive agreement statistics."""
    if providers is None:
        providers = [
            c.replace("cat_", "")
            for c in df.columns
            if c.startswith("cat_")
        ]

    total = len(df)
    if total == 0:
        return {}

    unanimous = int((df["agreement_level"] == "unanimous").sum())
    single_model = int((df["agreement_level"] == "single_model").sum())
    majority = int(df["agreement_level"].isin(["unanimous", "majority", "single_model"]).sum())
    no_agree = int((df["agreement_level"] == "no_agreement").sum())
    insufficient = int((df["agreement_level"] == "insufficient_data").sum())
    reliable = int(df["is_reliable"].sum()) if "is_reliable" in df.columns else 0
    diagrams = int(df["is_diagram"].sum()) if "is_diagram" in df.columns else 0
    non_diag = int(
        ((df["majority_category"] == "non_diagram") & df["is_reliable"]).sum()
    ) if "is_reliable" in df.columns else 0
    unreliable = total - reliable

    stats: dict = {
        "total_classified": total,
        "unanimous_rate": round(unanimous / total, 4),
        "single_model_count": single_model,
        "majority_rate": round(majority / total, 4),
        "mean_reliability": round(df["reliability_score"].mean(), 4) if "reliability_score" in df.columns else 0.0,
        "reliable_count": reliable,
        "diagram_count": diagrams,
        "non_diagram_count": non_diag,
        "unreliable_count": unreliable,
        "insufficient_data_count": insufficient,
    }

    # Agreement by subject
    if "subject" in df.columns:
        by_subj = {}
        for subj, group in df.groupby("subject"):
            n = len(group)
            una = (group["agreement_level"] == "unanimous").sum()
            by_subj[subj] = round(una / n, 4) if n else 0.0
        stats["agreement_by_subject"] = by_subj

    # Agreement by category
    if "majority_category" in df.columns:
        by_cat = {}
        valid = df[df["majority_category"] != "unknown"]
        for cat, group in valid.groupby("majority_category"):
            n = len(group)
            una = (group["agreement_level"] == "unanimous").sum()
            by_cat[cat] = round(una / n, 4) if n else 0.0
        stats["agreement_by_category"] = by_cat

    # Category distribution (reliable only)
    if "is_reliable" in df.columns and "majority_category" in df.columns:
        rel = df[df["is_reliable"]]
        stats["category_distribution"] = rel["majority_category"].value_counts().to_dict()

    # Reliability distribution
    if "reliability_score" in df.columns:
        scores = df["reliability_score"]
        stats["reliability_distribution"] = {
            "high": int((scores >= 0.8).sum()),
            "medium": int(((scores >= 0.5) & (scores < 0.8)).sum()),
            "low": int((scores < 0.5).sum()),
        }

    # Pairwise agreement
    if len(providers) >= 2:
        pairwise = {}
        for i, p1 in enumerate(providers):
            for p2 in providers[i + 1:]:
                c1 = f"cat_{p1}"
                c2 = f"cat_{p2}"
                if c1 in df.columns and c2 in df.columns:
                    valid_mask = (df[c1] != "unknown") & (df[c2] != "unknown")
                    valid_rows = df[valid_mask]
                    if len(valid_rows) > 0:
                        agree = (valid_rows[c1] == valid_rows[c2]).sum()
                        pairwise[f"{p1}_vs_{p2}"] = round(agree / len(valid_rows), 4)
        stats["pairwise_agreement"] = pairwise

    # Boolean decision distribution
    if "boolean_decision" in df.columns:
        bool_dist = df["boolean_decision"].value_counts().to_dict()
        stats["boolean_decision_distribution"] = bool_dist

        # How often do booleans agree with model's category?
        if "majority_category" in df.columns:
            model_says_diagram = df["majority_category"] != "non_diagram"
            bools_say_diagram = df["boolean_decision"] == "diagram"
            bools_say_non = df["boolean_decision"] == "non_diagram"

            # Cases where booleans overrode the model
            overrides_to_diagram = int(((~model_says_diagram) & bools_say_diagram).sum())
            overrides_to_non = int((model_says_diagram & bools_say_non).sum())
            stats["boolean_overrides"] = {
                "model_said_non_diagram_but_booleans_said_diagram": overrides_to_diagram,
                "model_said_diagram_but_booleans_said_non_diagram": overrides_to_non,
            }

    # Print summary
    print("=" * 60)
    print("CLASSIFICATION & RELIABILITY STATISTICS")
    print("=" * 60)
    print(f"  Total classified:       {total}")
    if single_model:
        print(f"  Single-model classified: {single_model}")
    if unanimous:
        print(f"  Unanimous agreement:    {unanimous} ({stats['unanimous_rate']:.1%})")
    print(f"  Reliable classified:    {majority} ({stats['majority_rate']:.1%})")
    if no_agree:
        print(f"  No agreement:           {no_agree}")
    if insufficient:
        print(f"  Insufficient data:      {insufficient}")
    print(f"  Mean reliability:       {stats['mean_reliability']:.4f}")
    print(f"  Reliable images:        {reliable}")
    print(f"  Diagrams:               {diagrams}")
    print(f"  Non-diagrams:           {non_diag}")
    print(f"  Unreliable (flagged):   {unreliable}")

    if "pairwise_agreement" in stats:
        print("\n  Pairwise agreement:")
        for pair, rate in stats["pairwise_agreement"].items():
            print(f"    {pair:<25} {rate:.1%}")

    if "agreement_by_subject" in stats:
        print("\n  Agreement by subject:")
        for subj, rate in sorted(stats["agreement_by_subject"].items(), key=lambda x: -x[1]):
            print(f"    {subj:<35} {rate:.1%}")

    if "category_distribution" in stats:
        print("\n  Category distribution (reliable):")
        for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
            print(f"    {cat:<30} {count}")

    if "reliability_distribution" in stats:
        rd = stats["reliability_distribution"]
        print(f"\n  Reliability distribution:")
        print(f"    High (>=0.8):  {rd['high']}")
        print(f"    Medium (0.5-0.8): {rd['medium']}")
        print(f"    Low (<0.5):    {rd['low']}")

    if "boolean_decision_distribution" in stats:
        print(f"\n  Boolean-based decisions:")
        for decision, count in stats["boolean_decision_distribution"].items():
            print(f"    {decision:<20} {count}")

    if "boolean_overrides" in stats:
        bo = stats["boolean_overrides"]
        print(f"\n  Boolean overrides:")
        print(f"    Booleans promoted to diagram:     {bo['model_said_non_diagram_but_booleans_said_diagram']}")
        print(f"    Booleans demoted to non-diagram:  {bo['model_said_diagram_but_booleans_said_non_diagram']}")

    print("=" * 60)
    return stats
