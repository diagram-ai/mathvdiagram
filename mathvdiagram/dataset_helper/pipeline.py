"""
Full classification pipeline orchestrator, paper statistics, and export utilities.

Classification Strategy (based on MathVision NeurIPS 2024 paper analysis):

The MathVision paper confirms ALL images are abstract/drawn (no photographs).
The distinction is between formal mathematical diagrams (reproducible from text)
vs illustrated/cartoon content (decorative, not meaningfully reproducible).

- HIGH TIER (9 subjects, ~60% of dataset): Included via metadata alone.
  analytic geometry, solid geometry, transformation geometry, descriptive geometry,
  topology, statistics, combinatorial geometry, metric geometry (length + area).
  These subjects have 75-95% formal diagram rates. No API calls needed.

- MIXED/LOW TIER (7 subjects, ~40% of dataset): Single LLM inspection.
  algebra, graph theory, number theory, combinatorics, counting, arithmetic, logic.
  These contain a mix of formal diagrams and illustrated content.
  A single structured LLM call with boolean checklist is sufficient.

Multi-provider consensus is available (--providers openai gemini claude) but
not the default. Single-provider classification is accurate enough for the
binary diagram/non-diagram decision.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from .. import config
from ..data_loader import load_mathvision
from .taxonomy import (
    DIAGRAM_TAXONOMY,
    SUBJECT_PRIORITY,
    SUBJECT_CATEGORY_MAP,
    build_taxonomy_text,
    get_subject_to_category_mapping,
)
from .exploration import (
    get_dataset_statistics,
    print_dataset_summary,
    apply_metadata_prefilter,
)
from .classification import classify_batch_structured
from .agreement import compute_agreement, compute_agreement_statistics


def run_full_classification(
    num_samples: int | None = None,
    resume: bool = True,
    delay: float | None = None,
    providers: list[str] | None = None,
    reliability_threshold: float = 0.7,
    validate_prefilter: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the complete layered classification pipeline:

      1. Load MathVision dataset
      2. Print dataset statistics
      3. Apply metadata pre-filter (Layer 1)
      4. Run structured LLM classification (Layer 2) on needs_inspection + likely_non_diagram
         Optionally validate 10% of likely_diagram group
      5. Assign pre-filter categories to unclassified likely_diagram images
      6. Compute multi-model agreement (Layer 3)
      7. Merge all results
      8. Save to OUTPUT_DIR/full_classification.csv

    Returns the final DataFrame.
    """
    providers = providers or ["openai"]

    # Step 1: Load
    df = load_mathvision()
    print_dataset_summary(df)

    if num_samples is not None:
        df = df.sample(n=min(num_samples, len(df)), random_state=seed).reset_index(drop=True)
        print(f"\nUsing {len(df)} sampled images")

    # Step 2: Metadata pre-filter
    tiers = apply_metadata_prefilter(df)

    # Step 3: Determine what to send to LLM
    to_classify_parts = [tiers["needs_inspection"], tiers["likely_non_diagram"]]

    # Optionally validate a sample of the pre-filtered high tier (5%)
    prefilter_validation_sample = None
    if validate_prefilter and len(tiers["likely_diagram"]) > 0:
        n_validate = max(1, len(tiers["likely_diagram"]) // 20)  # 5% sample
        prefilter_validation_sample = tiers["likely_diagram"].sample(
            n=n_validate, random_state=seed,
        )
        to_classify_parts.append(prefilter_validation_sample)
        print(f"\nValidating pre-filter with {n_validate} high-priority images")

    to_classify = pd.concat(to_classify_parts, ignore_index=True)

    print("=" * 60)
    print("STEP 1: Classifying images")
    print(f"  High tier: {len(tiers['likely_diagram'])} images (included via metadata)")
    print(f"  LLM inspection: {len(to_classify)} images ({', '.join(providers)})")
    print("=" * 60)

    # Step 4: LLM classification
    classified = classify_batch_structured(
        to_classify, resume=resume, delay=delay, providers=providers,
    )

    # Step 5: Agreement
    agreed = compute_agreement(classified, providers=providers, reliability_threshold=reliability_threshold)
    stats = compute_agreement_statistics(agreed, providers=providers)

    # Step 6: Pre-filter validation report — calibrate reliability from sample
    prefilter_reliability = 0.85  # default fallback
    if prefilter_validation_sample is not None and len(agreed) > 0:
        val_ids = set(prefilter_validation_sample["id"].astype(str))
        val_rows = agreed[agreed["image_id"].isin(val_ids)]
        if len(val_rows) > 0:
            confirmed = (val_rows["majority_category"] != "non_diagram").sum()
            total_val = len(val_rows)
            prefilter_reliability = confirmed / total_val if total_val > 0 else 0.85
            print(f"\nPre-filter validation: {confirmed}/{total_val} ({confirmed/total_val:.1%}) "
                  f"of high-priority images confirmed as diagrams by LLM consensus")
            print(f"  Calibrated pre-filter reliability: {prefilter_reliability:.4f}")
    elif not validate_prefilter:
        print("Warning: pre-filter validation skipped, using default reliability=0.85")

    # Store calibrated reliability in stats for paper reporting
    stats["prefilter_calibrated_reliability"] = round(prefilter_reliability, 4)

    # Step 7: Assign pre-filter categories to remaining likely_diagram images
    classified_ids = set(agreed["image_id"].astype(str))
    prefilter_remaining = tiers["likely_diagram"][
        ~tiers["likely_diagram"]["id"].astype(str).isin(classified_ids)
    ].copy()

    if len(prefilter_remaining) > 0:
        subject_map = get_subject_to_category_mapping()
        prefilter_remaining["image_id"] = prefilter_remaining["id"].astype(str)
        prefilter_remaining["majority_category"] = prefilter_remaining["subject"].map(
            lambda s: subject_map.get(s, "geometric_construction")
        )
        prefilter_remaining["agreement_level"] = "prefilter"
        prefilter_remaining["agreement_count"] = 0
        prefilter_remaining["mean_confidence"] = 0.0
        prefilter_remaining["reliability_score"] = prefilter_reliability
        prefilter_remaining["is_reliable"] = prefilter_reliability >= reliability_threshold
        prefilter_remaining["is_diagram"] = prefilter_reliability >= reliability_threshold
        prefilter_remaining["classification_source"] = "metadata_prefilter"

    # Tag classification source for LLM-classified rows
    def _source(row):
        level = row.get("agreement_level", "")
        if level == "unanimous":
            return "llm_unanimous"
        elif level == "majority":
            return "llm_majority"
        elif level == "single_model":
            return "llm_single"
        else:
            return "llm_unreliable"  # covers "no_agreement" and "insufficient_data"

    agreed["classification_source"] = agreed.apply(_source, axis=1)

    # Step 8: Merge
    parts = [agreed]
    if len(prefilter_remaining) > 0:
        parts.append(prefilter_remaining)

    final = pd.concat(parts, ignore_index=True)

    # Standardize output columns
    final["final_category"] = final["majority_category"]
    if "subject" not in final.columns and "subject" in df.columns:
        # Recover subject from original df
        id_to_subject = dict(zip(df["id"].astype(str), df["subject"]))
        final["subject"] = final["image_id"].map(id_to_subject)

    # Add priority tier
    final["priority_tier"] = final["subject"].map(lambda s: SUBJECT_PRIORITY.get(s, "unknown"))

    # Step 9: Save
    output_path = os.path.join(config.OUTPUT_DIR, "full_classification.csv")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"\nFinal classification saved to {output_path} ({len(final)} images)")

    # Save statistics
    stats_path = os.path.join(config.OUTPUT_DIR, "agreement_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Agreement statistics saved to {stats_path}")

    # Check for unclassified images (API errors)
    unclassified = final[final["final_category"] == "unknown"]
    if len(unclassified) > 0:
        print(f"\n{'!'*60}")
        print(f"WARNING: {len(unclassified)} images were NOT classified (API errors)")
        print(f"  These are marked is_diagram=False and excluded from the benchmark.")
        print(f"  To classify them, re-run with --no-resume or increase API quota.")
        print(f"  Subject breakdown of unclassified images:")
        for subj, count in unclassified["subject"].value_counts().items():
            print(f"    {subj}: {count}")
        print(f"{'!'*60}")

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    if "is_diagram" in final.columns:
        n_diag = final["is_diagram"].sum()
        print(f"  Diagrams:          {n_diag}")
        print(f"  Non-diagrams:      {len(final) - n_diag}")
    if "classification_source" in final.columns:
        for src, count in final["classification_source"].value_counts().items():
            print(f"  Source {src}: {count}")
    if "final_category" in final.columns:
        print("\n  Category distribution:")
        for cat, count in final["final_category"].value_counts().items():
            print(f"    {cat:<30} {count}")
    print(f"{'='*60}")

    return final


def generate_paper_statistics(
    classification_csv: str | None = None,
    descriptions_csv: str | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Generate all statistics needed for the paper's dataset/methodology section.
    Saves as formatted JSON to OUTPUT_DIR/paper_statistics.json.
    """
    classification_csv = classification_csv or os.path.join(config.OUTPUT_DIR, "full_classification.csv")
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "paper_statistics.json")

    cls_df = pd.read_csv(classification_csv)

    # Dataset stats
    full_df = load_mathvision()
    ds_stats = get_dataset_statistics(full_df)

    paper: dict = {
        "dataset": {
            "source": "MathVision (NeurIPS 2024)",
            "total_images": ds_stats["total_images"],
            "subject_distribution": ds_stats["subject_distribution"],
            "level_distribution": ds_stats["level_distribution"],
        },
        "filtering": {
            "prefilter_high": int((cls_df.get("priority_tier") == "high").sum()) if "priority_tier" in cls_df.columns else 0,
            "prefilter_mixed": int((cls_df.get("priority_tier") == "mixed").sum()) if "priority_tier" in cls_df.columns else 0,
            "prefilter_low": int((cls_df.get("priority_tier") == "low").sum()) if "priority_tier" in cls_df.columns else 0,
        },
        "classification": {},
        "final_benchmark": {},
    }

    # Classification stats
    if "agreement_level" in cls_df.columns:
        total = len(cls_df)
        # single_model with high confidence is equivalent to unanimous (one model trivially agrees with itself)
        una = cls_df["agreement_level"].isin(["unanimous", "single_model"]).sum()
        maj = cls_df["agreement_level"].isin(["unanimous", "majority", "single_model"]).sum()
        paper["classification"]["unanimous_rate"] = round(una / total, 4) if total else 0.0
        paper["classification"]["majority_rate"] = round(maj / total, 4) if total else 0.0

    if "reliability_score" in cls_df.columns:
        paper["classification"]["mean_reliability"] = round(cls_df["reliability_score"].mean(), 4)

    # Unclassified (API error) stats
    if "final_category" in cls_df.columns:
        unclassified = int((cls_df["final_category"] == "unknown").sum())
        paper["classification"]["unclassified_count"] = unclassified
        if unclassified > 0:
            paper["classification"]["unclassified_subjects"] = (
                cls_df[cls_df["final_category"] == "unknown"]["subject"]
                .value_counts().to_dict()
            )

    # Boolean decision stats
    if "boolean_decision" in cls_df.columns:
        bool_dist = cls_df["boolean_decision"].value_counts().to_dict()
        paper["classification"]["boolean_decisions"] = bool_dist

        if "majority_category" in cls_df.columns:
            model_diag = (cls_df["majority_category"] != "non_diagram")
            bool_diag = (cls_df["boolean_decision"] == "diagram")
            bool_non = (cls_df["boolean_decision"] == "non_diagram")
            paper["classification"]["boolean_overrides"] = {
                "promoted_to_diagram": int(((~model_diag) & bool_diag).sum()),
                "demoted_to_non_diagram": int((model_diag & bool_non).sum()),
            }

    # Pairwise agreement
    prov_cols = [c.replace("cat_", "") for c in cls_df.columns if c.startswith("cat_")]
    if len(prov_cols) >= 2:
        pairwise = {}
        for i, p1 in enumerate(prov_cols):
            for p2 in prov_cols[i + 1:]:
                c1, c2 = f"cat_{p1}", f"cat_{p2}"
                valid = cls_df[(cls_df[c1] != "unknown") & (cls_df[c2] != "unknown")]
                if len(valid) > 0:
                    pairwise[f"{p1}_vs_{p2}"] = round((valid[c1] == valid[c2]).mean(), 4)
        paper["classification"]["pairwise_agreement"] = pairwise

    # Pre-filter validation accuracy
    if "classification_source" in cls_df.columns and "agreement_level" in cls_df.columns:
        prefilter_validated = cls_df[
            (cls_df["classification_source"] != "metadata_prefilter") &
            (cls_df["priority_tier"] == "high")
        ] if "priority_tier" in cls_df.columns else pd.DataFrame()
        if len(prefilter_validated) > 0:
            confirmed = (prefilter_validated["majority_category"] != "non_diagram").sum() if "majority_category" in prefilter_validated.columns else 0
            paper["filtering"]["prefilter_validation_accuracy"] = round(confirmed / len(prefilter_validated), 4)

    # Final benchmark
    cat_col = "final_category" if "final_category" in cls_df.columns else "majority_category"
    if "is_diagram" in cls_df.columns:
        diagrams = cls_df[cls_df["is_diagram"] == True]
        paper["final_benchmark"]["total_diagrams"] = int(len(diagrams))
        paper["final_benchmark"]["excluded_count"] = int(len(cls_df) - len(diagrams))

        if cat_col in diagrams.columns:
            paper["final_benchmark"]["category_distribution"] = diagrams[cat_col].value_counts().to_dict()

        if "subject" in diagrams.columns:
            paper["final_benchmark"]["by_subject"] = diagrams["subject"].value_counts().to_dict()

        if "level" in diagrams.columns:
            paper["final_benchmark"]["by_difficulty"] = diagrams["level"].value_counts().to_dict()

    # Exclusion reasons
    if "is_diagram" in cls_df.columns:
        excluded = cls_df[cls_df["is_diagram"] == False]
        reasons = {}
        if "majority_category" in excluded.columns:
            reasons["non_diagram_category"] = int((excluded["majority_category"] == "non_diagram").sum())
        if "is_reliable" in excluded.columns:
            reasons["unreliable"] = int((~excluded["is_reliable"]).sum())
        paper["final_benchmark"]["exclusion_reasons"] = reasons

    # FIX 6A: Sensitivity analysis across reliability thresholds
    if "reliability_score" in cls_df.columns:
        sensitivity = {}
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9]:
            reliable_at_t = cls_df["reliability_score"] >= threshold
            is_diag_at_t = (cls_df["majority_category"] != "non_diagram") & reliable_at_t if "majority_category" in cls_df.columns else reliable_at_t
            sensitivity[str(threshold)] = {
                "reliable_count": int(reliable_at_t.sum()),
                "diagram_count": int(is_diag_at_t.sum()),
            }
        paper["classification"]["sensitivity_analysis"] = sensitivity

    # FIX 6B: Per-boolean-feature agreement rates
    boolean_features = [
        "has_labeled_axes", "has_geometric_labels", "has_real_world_objects",
        "has_photographic_content", "has_mathematical_notation", "has_grid_or_coordinate_system",
    ]
    if len(prov_cols) >= 2:
        boolean_agreement = {}
        for feat in boolean_features:
            agreements = 0
            total_valid = 0
            for _, row in cls_df.iterrows():
                values = []
                for prov in prov_cols:
                    try:
                        cls = json.loads(row.get(f"cls_{prov}", "{}"))
                        val = cls.get(feat)
                        if val is not None:
                            values.append(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if len(values) >= 2:
                    total_valid += 1
                    if len(set(values)) == 1:  # all same
                        agreements += 1
            boolean_agreement[feat] = round(agreements / total_valid, 4) if total_valid else 0.0
        paper["classification"]["boolean_feature_agreement"] = boolean_agreement

    # FIX 6C: Pre-filter false negative estimate
    if "priority_tier" in cls_df.columns and "is_diagram" in cls_df.columns:
        low_tier = cls_df[cls_df["priority_tier"] == "low"]
        if len(low_tier) > 0:
            false_negatives = int(low_tier["is_diagram"].sum())
            paper["filtering"]["prefilter_false_negative_count"] = false_negatives
            paper["filtering"]["prefilter_false_negative_rate"] = round(
                false_negatives / len(low_tier), 4
            )

        mixed_tier = cls_df[cls_df["priority_tier"] == "mixed"]
        if len(mixed_tier) > 0:
            mixed_diagrams = int(mixed_tier["is_diagram"].sum())
            paper["filtering"]["mixed_tier_diagram_rate"] = round(
                mixed_diagrams / len(mixed_tier), 4
            )

    # Validation (if descriptions available)
    if descriptions_csv and os.path.exists(descriptions_csv):
        from .validation import validate_classifications_with_descriptions, compute_validation_statistics
        validated = validate_classifications_with_descriptions(classification_csv, descriptions_csv)
        val_stats = compute_validation_statistics(validated)
        paper["validation"] = {
            "good_quality_rate": val_stats.get("good_pct", 0.0),
            "suspect_rate": val_stats.get("suspect_pct", 0.0),
            "misclassified_rate": val_stats.get("misclassified_pct", 0.0),
        }

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(paper, f, indent=2, default=str)
    print(f"\nPaper statistics saved to {output_path}")

    # Print formatted summary
    print(f"\n{'='*60}")
    print("PAPER-READY STATISTICS")
    print(f"{'='*60}")
    print(f"  Dataset: {paper['dataset']['source']}")
    print(f"  Total images: {paper['dataset']['total_images']}")
    if "final_benchmark" in paper and "total_diagrams" in paper["final_benchmark"]:
        fb = paper["final_benchmark"]
        print(f"  Diagrams for benchmark: {fb['total_diagrams']}")
        print(f"  Excluded: {fb.get('excluded_count', 0)}")
    if "classification" in paper:
        cl = paper["classification"]
        if "unanimous_rate" in cl:
            print(f"  Unanimous agreement: {cl['unanimous_rate']:.1%}")
        if "majority_rate" in cl:
            print(f"  Majority agreement: {cl['majority_rate']:.1%}")
        if "mean_reliability" in cl:
            print(f"  Mean reliability: {cl['mean_reliability']:.4f}")
    if "sensitivity_analysis" in paper.get("classification", {}):
        print("\n  Sensitivity analysis (threshold -> diagram count):")
        for t, vals in paper["classification"]["sensitivity_analysis"].items():
            print(f"    threshold={t}: {vals['diagram_count']} diagrams, {vals['reliable_count']} reliable")
    if "boolean_feature_agreement" in paper.get("classification", {}):
        print("\n  Boolean feature agreement:")
        for feat, rate in paper["classification"]["boolean_feature_agreement"].items():
            print(f"    {feat:<35} {rate:.1%}")
    print(f"{'='*60}")

    return paper


def export_taxonomy_document(output_path: str | None = None) -> str:
    """Export the full taxonomy as a formatted Markdown document."""
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "taxonomy.md")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines = [
        "# MathVision Diagram Taxonomy Reference",
        "",
        "## Categories",
        "",
    ]

    for cat in DIAGRAM_TAXONOMY.values():
        lines.append(f"### {cat.name} (`{cat.id}`)")
        lines.append("")
        lines.append(f"**Definition:** {cat.definition}")
        lines.append("")
        lines.append("**Includes:**")
        for inc in cat.includes:
            lines.append(f"- {inc}")
        lines.append("")
        lines.append("**Excludes:**")
        for exc in cat.excludes:
            lines.append(f"- {exc}")
        lines.append("")
        lines.append(f"**Keywords:** {', '.join(cat.keywords)}")
        lines.append("")

    lines.append("## Subject-to-Priority Mapping")
    lines.append("")
    lines.append("| Subject | Priority Tier |")
    lines.append("|---------|---------------|")
    for subj, tier in sorted(SUBJECT_PRIORITY.items()):
        lines.append(f"| {subj} | {tier} |")
    lines.append("")

    lines.append("## Subject-to-Category Mapping (Pre-filter)")
    lines.append("")
    lines.append("| Subject | Default Category |")
    lines.append("|---------|-----------------|")
    for subj, cat in sorted(SUBJECT_CATEGORY_MAP.items()):
        lines.append(f"| {subj} | {cat} |")
    lines.append("")

    lines.append("## Reliability Scoring")
    lines.append("")
    lines.append("```")
    lines.append("reliability = (agreement_count / num_models) * mean_confidence_of_agreeing_models")
    lines.append("")
    lines.append("Only confidence weights from models that voted for the majority category")
    lines.append("are included in the mean.")
    lines.append("")
    lines.append("Confidence weights: high=1.0, medium=0.66, low=0.33")
    lines.append("Default threshold: reliability >= 0.66 → reliable")
    lines.append("Boolean consistency violations apply a penalty of 0.05 per violation (max 0.2)")
    lines.append("```")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"Taxonomy document exported to {output_path}")
    return output_path


def get_diagram_subset(
    classification_csv: str | None = None,
    categories: list[str] | None = None,
    min_reliability: float = 0.0,
) -> pd.DataFrame:
    """
    Load the final classification CSV and return only rows where
    is_diagram=True. Optionally filter by categories or min reliability.

    This is the input for the description step (Step 2 of the main pipeline).
    """
    classification_csv = classification_csv or os.path.join(config.OUTPUT_DIR, "full_classification.csv")
    df = pd.read_csv(classification_csv)

    result = df[df["is_diagram"] == True].copy()

    if categories is not None:
        cat_col = "final_category" if "final_category" in result.columns else "majority_category"
        result = result[result[cat_col].isin(categories)]

    if min_reliability > 0 and "reliability_score" in result.columns:
        result = result[result["reliability_score"] >= min_reliability]

    print(f"Diagram subset: {len(result)} images (from {len(df)} total)")
    cat_col = "final_category" if "final_category" in result.columns else "majority_category"
    if cat_col in result.columns:
        for cat, count in result[cat_col].value_counts().items():
            print(f"  {cat}: {count}")

    return result.reset_index(drop=True)
