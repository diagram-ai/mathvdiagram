"""
dataset_helper — taxonomy-aware classification, multi-model agreement, and validation tools.

No human annotation required. Uses multi-provider LLM consensus and
reliability scoring for classification validation.
"""

from .taxonomy import (
    TaxonomyCategory,
    DiagramCategory,
    DIAGRAM_TAXONOMY,
    VALID_CATEGORIES,
    SUBJECT_PRIORITY,
    SUBJECT_CATEGORY_MAP,
    build_taxonomy_text,
    get_subject_to_category_mapping,
)
from .exploration import (
    get_dataset_statistics,
    print_dataset_summary,
    sample_by_subject,
    get_annotation_sample,
    apply_metadata_prefilter,
)
from .classification import (
    parse_classification_response,
    classify_single_image_structured,
    classify_batch_structured,
)
from .agreement import (
    CONFIDENCE_WEIGHTS,
    CATEGORY_BOOLEAN_RULES,
    check_boolean_category_consistency,
    classify_from_booleans,
    compute_reliability_score,
    compute_agreement,
    compute_agreement_statistics,
)
from .validation import (
    detect_quality_signals,
    validate_classifications_with_descriptions,
    compute_validation_statistics,
    compute_independent_metrics,
    run_filter_ablation,
)
from .pipeline import (
    run_full_classification,
    generate_paper_statistics,
    export_taxonomy_document,
    get_diagram_subset,
)
from .report import generate_classification_report

__all__ = [
    # taxonomy
    "TaxonomyCategory",
    "DiagramCategory",
    "DIAGRAM_TAXONOMY",
    "VALID_CATEGORIES",
    "SUBJECT_PRIORITY",
    "SUBJECT_CATEGORY_MAP",
    "build_taxonomy_text",
    "get_subject_to_category_mapping",
    # exploration
    "get_dataset_statistics",
    "print_dataset_summary",
    "sample_by_subject",
    "get_annotation_sample",
    "apply_metadata_prefilter",
    # classification
    "parse_classification_response",
    "classify_single_image_structured",
    "classify_batch_structured",
    # agreement
    "CONFIDENCE_WEIGHTS",
    "CATEGORY_BOOLEAN_RULES",
    "check_boolean_category_consistency",
    "classify_from_booleans",
    "compute_reliability_score",
    "compute_agreement",
    "compute_agreement_statistics",
    # validation
    "detect_quality_signals",
    "validate_classifications_with_descriptions",
    "compute_validation_statistics",
    "compute_independent_metrics",
    "run_filter_ablation",
    # pipeline
    "run_full_classification",
    "generate_paper_statistics",
    "export_taxonomy_document",
    "get_diagram_subset",
    # report
    "generate_classification_report",
]
