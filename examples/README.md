# Examples

Runnable scripts for the full MathVDiagram workflow. Run them from the
repository root with the package installed (`pip install -e ".[benchmark]"`) and
the relevant API keys exported (see [`.env.example`](../.env.example)).

## Stage 1: Prompt curation

Build the curated `concise_prompts.csv` from the MathVision dataset (classify,
describe with an ensemble of VLMs, aggregate). This is driven by the package
CLI rather than a shell script:

```bash
# Full curation pipeline
python -m mathvdiagram.pipeline

# Quick smoke test on a few images
python -m mathvdiagram.pipeline --num-samples 10
```

Review the result and record exclusions:

```bash
python -m mathvdiagram.benchmark.curate
```

## Stage 2: Benchmark

| Script | What it does |
|--------|--------------|
| [`run_experiment.sh`](run_experiment.sh) | Runs generation -> evaluation -> per-model report for every model, then the cross-model comparison. Honors `MODELS` and `WORKERS` env vars. |
| [`setup_gcp.sh`](setup_gcp.sh) | Provisions a fresh GCP VM (TeX Live, poppler, Cairo, venv, dependencies) for a full run. |

```bash
# Everything, all models
bash examples/run_experiment.sh

# A subset, more workers
MODELS="deepseek-v3 gpt-5.4" WORKERS=10 bash examples/run_experiment.sh
```

Individual stages can also be invoked directly:

```bash
python -m mathvdiagram.benchmark.generate --model deepseek-v3 --limit 20 --workers 4
python -m mathvdiagram.benchmark.evaluate --model deepseek-v3
python -m mathvdiagram.benchmark.report   --model deepseek-v3
python -m mathvdiagram.benchmark.compare
python -m mathvdiagram.benchmark.paper_figures
python -m mathvdiagram.benchmark.hf_upload --dry-run
```

See [`mathvdiagram/benchmark/README.md`](../mathvdiagram/benchmark/README.md)
for the full description of each module and its inputs/outputs.
