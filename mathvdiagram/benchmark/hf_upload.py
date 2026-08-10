"""
hf_upload.py -- Build and upload the MathVision Diagram Benchmark to HuggingFace.

Assembles three dataset configs (full / common_subset / ground_truth) from the
generated images, ground truth, and evaluation results, then pushes them plus
supplementary per-model eval JSONs.

Paths are resolved relative to the current working directory (repo root).

Run with::

    export HF_TOKEN=hf_xxx
    python -m mathvdiagram.benchmark.hf_upload
    python -m mathvdiagram.benchmark.hf_upload --dry-run
"""

import argparse
import csv
import os
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Value
from huggingface_hub import HfApi

from .models import MODEL_ORDER

DEFAULT_REPO_ID = "mathdiagrams/MathVisionBenchmark"

PROMPTS_CSV = Path("data") / "concise_prompts.csv"
GROUND_TRUTH = Path("data") / "ground_truth"
OUTPUTS = Path("outputs")
FULL_RESULTS = Path("paper") / "data" / "full_results.csv"
COMMON_SUBSET = Path("paper") / "data" / "common_subset.csv"
SUMMARY_STATS = Path("paper") / "data" / "summary_stats.json"

PAPER_MODELS = list(MODEL_ORDER)

FEATURES = Features({
    "image_id": Value("string"),
    "model": Value("string"),
    "model_name": Value("string"),
    "category": Value("string"),
    "concise_prompt": Value("string"),
    "original_question": Value("string"),
    "code_language": Value("string"),
    "ground_truth_image": HFImage(),
    "generated_image": HFImage(),
    "dists": Value("float32"),
    "clip_sim": Value("float32"),
    "edge_iou": Value("float32"),
    "edge_f1": Value("float32"),
})

GT_FEATURES = Features({
    "image_id": Value("string"),
    "category": Value("string"),
    "concise_prompt": Value("string"),
    "original_question": Value("string"),
    "ground_truth_image": HFImage(),
})


def load_prompts():
    """Load prompts CSV into {image_id: row}."""
    prompts = {}
    with open(PROMPTS_CSV) as f:
        for row in csv.DictReader(f):
            prompts[row["image_id"]] = row
    return prompts


def generate_rows(results_csv, prompts):
    """Yield one record per row in a results CSV, skipping missing images."""
    skipped = 0
    yielded = 0
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            image_id = row["image_id"]

            gen_path = OUTPUTS / model / f"{image_id}.png"
            gt_path = GROUND_TRUTH / f"{image_id}.png"

            if not gen_path.exists() or not gt_path.exists():
                skipped += 1
                continue

            if image_id not in prompts:
                skipped += 1
                continue

            prompt_info = prompts[image_id]
            yielded += 1
            yield {
                "image_id": image_id,
                "model": model,
                "model_name": row.get("model_name", model),
                "category": row.get("category", "unknown"),
                "concise_prompt": prompt_info.get("concise_prompt", ""),
                "original_question": prompt_info.get("question", ""),
                "code_language": row.get("code_language", "unknown"),
                "ground_truth_image": str(gt_path),
                "generated_image": str(gen_path),
                "dists": float(row["dists"]),
                "clip_sim": float(row["clip_sim"]),
                "edge_iou": float(row["edge_iou"]),
                "edge_f1": float(row["edge_f1"]),
            }

    print(f"  Yielded {yielded} rows, skipped {skipped} (missing images)")


def generate_gt_rows(prompts):
    """Yield one record per curated prompt with ground truth image."""
    for image_id, info in prompts.items():
        gt_path = GROUND_TRUTH / f"{image_id}.png"
        if not gt_path.exists():
            continue
        yield {
            "image_id": image_id,
            "category": info.get("category", "unknown"),
            "concise_prompt": info.get("concise_prompt", ""),
            "original_question": info.get("question", ""),
            "ground_truth_image": str(gt_path),
        }


def validate_prerequisites():
    """Check all required files and directories exist before starting."""
    errors = []

    if not PROMPTS_CSV.exists():
        errors.append(f"Missing: {PROMPTS_CSV}")
    if not GROUND_TRUTH.exists():
        errors.append(f"Missing: {GROUND_TRUTH}")
    if not FULL_RESULTS.exists():
        errors.append(f"Missing: {FULL_RESULTS}")
    if not COMMON_SUBSET.exists():
        errors.append(f"Missing: {COMMON_SUBSET}")

    gt_count = len(list(GROUND_TRUTH.glob("*.png"))) if GROUND_TRUTH.exists() else 0
    print(f"Ground truth images: {gt_count}")

    for model in PAPER_MODELS:
        model_dir = OUTPUTS / model
        if not model_dir.exists():
            errors.append(f"Missing model dir: {model_dir}")
            continue
        png_count = len(list(model_dir.glob("*.png")))
        print(f"  {model}: {png_count} PNGs")
        if png_count == 0:
            errors.append(f"No PNGs in {model_dir}")

    if errors:
        print("\nFATAL -- cannot proceed:")
        for e in errors:
            print(f"  - {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target HuggingFace dataset repo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and build datasets without pushing")
    args = parser.parse_args()

    repo_id = args.repo_id

    print("=== Validating prerequisites ===")
    if not validate_prerequisites():
        return

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: Set HF_TOKEN environment variable")
        return

    prompts = load_prompts()
    print(f"\nLoaded {len(prompts)} prompts")

    print("\n=== Building 'full' dataset ===")
    full_ds = Dataset.from_generator(
        lambda: generate_rows(FULL_RESULTS, prompts),
        features=FEATURES,
    )
    print(f"  Full dataset: {len(full_ds)} rows")

    print("\n=== Building 'common_subset' dataset ===")
    common_ds = Dataset.from_generator(
        lambda: generate_rows(COMMON_SUBSET, prompts),
        features=FEATURES,
    )
    print(f"  Common subset: {len(common_ds)} rows")

    print("\n=== Building 'ground_truth' dataset ===")
    gt_ds = Dataset.from_generator(
        lambda: generate_gt_rows(prompts),
        features=GT_FEATURES,
    )
    print(f"  Ground truth: {len(gt_ds)} rows")

    if args.dry_run:
        print("\n=== Dry run -- skipping push ===")
        print("Sampling 3 rows from full dataset:")
        for i in range(min(3, len(full_ds))):
            row = full_ds[i]
            print(f"  [{i}] model={row['model']} id={row['image_id']} "
                  f"dists={row['dists']:.3f} clip={row['clip_sim']:.3f}")
        return

    print(f"\n=== Pushing to {repo_id} ===")

    print("Pushing 'full' config...")
    full_ds.push_to_hub(repo_id, config_name="full", split="test",
                        max_shard_size="500MB", token=token)

    print("Pushing 'common_subset' config...")
    common_ds.push_to_hub(repo_id, config_name="common_subset", split="test",
                          max_shard_size="500MB", token=token)

    print("Pushing 'ground_truth' config...")
    gt_ds.push_to_hub(repo_id, config_name="ground_truth", split="test",
                      max_shard_size="500MB", token=token)

    print("\n=== Uploading supplementary files ===")
    api = HfApi(token=token)

    if SUMMARY_STATS.exists():
        api.upload_file(
            path_or_fileobj=str(SUMMARY_STATS),
            path_in_repo="supplementary/summary_stats.json",
            repo_id=repo_id, repo_type="dataset",
        )
        print("  Uploaded summary_stats.json")

    for model in PAPER_MODELS:
        eval_file = OUTPUTS / model / "eval_results.json"
        if eval_file.exists():
            api.upload_file(
                path_or_fileobj=str(eval_file),
                path_in_repo=f"supplementary/eval_results/{model}.json",
                repo_id=repo_id, repo_type="dataset",
            )
            print(f"  Uploaded {model} eval_results.json")

    print(f"\n=== Done! Dataset at https://huggingface.co/datasets/{repo_id} ===")


if __name__ == "__main__":
    main()
