# mathvdiagram.benchmark

The benchmark half of MathVDiagram. It takes the curated `concise_prompts.csv`
produced by the [curation pipeline](../../README.md#stage-1-prompt-curation)
and measures how well models can redraw each diagram from its natural-language
description alone.

Install the optional dependencies first:

```bash
pip install "mathvdiagram[benchmark]"
```

Generation also needs a system TeX stack for the TikZ compiler
([TeX Live](https://www.tug.org/texlive) with `pdflatex` + `pdftoppm` from
[poppler](https://poppler.freedesktop.org)) and Cairo for the SVG compiler.

## Modules

| Module | Entry point | Purpose |
|--------|-------------|---------|
| `models` | (library) | Model registry, display names, and OpenAI-compatible client + generation helpers. |
| `generate` | `python -m mathvdiagram.benchmark.generate` | Prompt each model, detect the output format (TikZ / SVG / Python / image), compile to PNG. Writes `outputs/<model>/*.png` + `generation_log.json`. |
| `evaluate` | `python -m mathvdiagram.benchmark.evaluate` | Score generated vs. ground-truth images: DISTS, CLIP cosine similarity, Edge IoU, Edge F1 (per-pair) and CMMD (distribution-level). Writes `outputs/<model>/eval_results.json`. |
| `compare` | `python -m mathvdiagram.benchmark.compare` | Cross-model leaderboard (all-pairs + common subset) with Wilcoxon signed-rank + Holm-Bonferroni significance. Writes `outputs/comparison_report.html`. |
| `report` | `python -m mathvdiagram.benchmark.report` | Single-model HTML report with ground-truth-vs-generated side-by-sides. Writes `outputs/<model>/report.html`. |
| `paper_figures` | `python -m mathvdiagram.benchmark.paper_figures` | Publication figures (PDF), LaTeX tables, and CSV/JSON exports into `paper/`. |
| `curate` | `python -m mathvdiagram.benchmark.curate` | Local web UI to review prompts and record exclusions in `data/excluded_ids.txt`. |
| `hf_upload` | `python -m mathvdiagram.benchmark.hf_upload` | Build and push the HuggingFace dataset (three configs) + supplementary eval JSONs. |

## Inputs and outputs

```
data/concise_prompts.csv      # curated prompts (image_id, question, category, concise_prompt, ...)
data/ground_truth/<id>.png    # reference diagrams from MathVision (download separately; gitignored)
        |
   generate  ->  outputs/<model>/<id>.png + generation_log.json
        |
   evaluate  ->  outputs/<model>/eval_results.json
        |
   compare / report / paper_figures  ->  HTML + paper/ assets
        |
   hf_upload ->  HuggingFace dataset
```

`data/concise_prompts.csv` is the hand-off point from the curation pipeline: run
the curation `run_benchmarking_pipeline` to produce `output/concise_prompts.csv`,
review it with `curate`, then place the curated file at `data/concise_prompts.csv`.

## Quick start

```bash
# One model, a handful of prompts
python -m mathvdiagram.benchmark.generate --model deepseek-v3 --limit 20 --workers 4
python -m mathvdiagram.benchmark.evaluate --model deepseek-v3
python -m mathvdiagram.benchmark.report   --model deepseek-v3

# All models end-to-end (see examples/run_experiment.sh)
bash examples/run_experiment.sh
python -m mathvdiagram.benchmark.compare
python -m mathvdiagram.benchmark.paper_figures
```

## Metrics

- **DISTS** (lower is better): perceptual distance from deep features.
- **CLIP Similarity** (higher is better): cosine similarity of CLIP ViT-B/32 embeddings.
- **Edge IoU** / **Edge F1** (higher is better): overlap of Canny edge masks; well suited to line drawings.
- **CMMD** (lower is better): CLIP Maximum Mean Discrepancy at the distribution level (unbiased MMD, median-heuristic bandwidth).

## API keys

Each model reads its key from the environment (loaded from `.env`). Depending on
which models you run you may need: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
`GROQ_API_KEY`, and `OPENROUTER_API_KEY`. The HuggingFace upload needs `HF_TOKEN`.
