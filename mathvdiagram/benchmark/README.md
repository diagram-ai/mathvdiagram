# Stage 2: Benchmark

Once you have a curated `data/concise_prompts.csv` (produced by
[Stage 1](../README.md)), this subpackage runs the full evaluation pipeline:

| Step | Module | Output |
|------|--------|--------|
| Generate | `benchmark.generate` | Prompt each model; compile TikZ / SVG / Python (or take image output) to `outputs/<model>/*.png` |
| Evaluate | `benchmark.evaluate` | DISTS, CLIP similarity, Edge IoU, Edge F1, and CMMD -> `outputs/<model>/eval_results.json` |
| Compare | `benchmark.compare` | Cross-model leaderboard + Wilcoxon/Holm significance -> `outputs/comparison_report.html` |
| Report | `benchmark.report` | Per-model side-by-side HTML |
| Paper | `benchmark.paper_figures` | PDF figures, LaTeX tables, CSV/JSON exports into `paper/` |
| Curate | `benchmark.curate` | Local web UI to record `data/excluded_ids.txt` |
| Upload | `benchmark.hf_upload` | Build + push the HuggingFace dataset |

## Models

Code LLMs emit TikZ/SVG/Python that is compiled; image models return images
directly. The default suite:

| Model | Type | Provider |
|-------|------|----------|
| DeepSeek V3 / R1 | Code LLM | DeepSeek |
| GPT-5.4 | Code LLM | OpenAI |
| GPT-OSS-120B | Code LLM | Groq |
| Claude Opus 4.6 | Code LLM | OpenRouter |
| Gemini 3.1 Pro | Code LLM | OpenRouter |
| Qwen3.5-35B | Code LLM | OpenRouter |
| Llama 4 Maverick | Code LLM | OpenRouter |
| Kimi K2.5 | Code LLM | OpenRouter |
| Nano Banana 2 / Pro | Image Gen | OpenRouter |

## Metrics

- **DISTS** (lower is better): perceptual distance from deep features.
- **CLIP Similarity** (higher is better): cosine similarity of CLIP ViT-B/32 embeddings.
- **Edge IoU** / **Edge F1** (higher is better): overlap of Canny edge masks; well suited to line drawings.
- **CMMD** (lower is better): CLIP Maximum Mean Discrepancy (unbiased MMD, median-heuristic bandwidth).

## Usage

All commands run from the repository root.

```sh
# One model, a few prompts
python -m mathvdiagram.benchmark.generate --model deepseek-v3 --limit 20 --workers 4
python -m mathvdiagram.benchmark.evaluate --model deepseek-v3
python -m mathvdiagram.benchmark.report   --model deepseek-v3

# All models end-to-end, then compare + paper assets
bash examples/run_experiment.sh
python -m mathvdiagram.benchmark.compare
python -m mathvdiagram.benchmark.paper_figures

# Curate: review prompts and record exclusions
python -m mathvdiagram.benchmark.curate

# Publish to HuggingFace (dry-run first)
python -m mathvdiagram.benchmark.hf_upload --dry-run
python -m mathvdiagram.benchmark.hf_upload
```

## Output files

### Per-model (in `outputs/<model>/`)

| File | Description |
|------|-------------|
| `*.png` | Generated diagram images |
| `*.code` | Raw code returned by the model |
| `generation_log.json` | Per-prompt generation metadata (timing, format, errors) |
| `eval_results.json` | Per-image metric scores (DISTS, CLIP, Edge IoU/F1) + aggregate CMMD |

### Cross-model

| File | Description |
|------|-------------|
| `outputs/comparison_report.html` | Leaderboard with significance tests |
| `paper/figures/*.pdf` | Publication figures (distributions, compile rates, category heatmaps, correlations, significance, difficulty, radar, scatter) |
| `paper/tables/*.tex` | LaTeX tables (`tab1_leaderboard_all.tex`, `tab1_leaderboard_common.tex`, `tab2_categories.tex`) |
| `paper/data/` | Flat exports for reproducibility (`full_results.csv`, `common_subset.csv`, `summary_stats.json`) |
| `paper/reports/` | Per-model HTML reports (tracked via Git LFS) |

## Configuration

Benchmark-specific environment variables (in `.env`):

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `GROQ_API_KEY` | Groq API key |
| `OPENROUTER_API_KEY` | OpenRouter API key (Claude, Gemini, Qwen, Llama, Kimi, Nano Banana) |

## File details

| File | Purpose |
|------|---------|
| **`models.py`** | Model registry. `MODELS` dict with provider URLs, API key env vars, and model IDs. `get_client()` returns an OpenAI-compatible client for any provider. `generate_code()` / `generate_image()` send prompts to code LLMs or image generators. |
| **`generate.py`** | Generation driver. Extracts code blocks from LLM responses, detects format (TikZ/SVG/Python), compiles to PNG. Runs prompts in parallel with `ThreadPoolExecutor`. Caches outputs to skip completed work on re-runs. |
| **`evaluate.py`** | Scoring engine. Computes DISTS, CLIP similarity, Edge IoU/F1 per image pair, and CMMD across the full model run. Outputs `eval_results.json`. |
| **`compare.py`** | Cross-model comparison. Builds a leaderboard on the common subset of successfully generated images, runs pairwise Wilcoxon signed-rank tests with Holm correction, produces `comparison_report.html`. |
| **`report.py`** | Per-model HTML report with generation summary, overall metrics, per-category breakdown, and side-by-side ground truth vs. generated image comparisons. |
| **`paper_figures.py`** | Publication assets. Generates PDF figures (distributions, compile rates, category heatmaps, format analysis, correlations, significance matrices, radar charts, scatter plots), LaTeX tables, and CSV/JSON data exports into `paper/`. |
| **`curate.py`** | Local web UI served on `localhost:8000`. Browse prompts, view ground truth images, and record exclusion decisions in `data/excluded_ids.txt`. |
| **`hf_upload.py`** | HuggingFace dataset builder. Assembles `full`, `common_subset`, and `ground_truth` configs from outputs and pushes to the Hub along with supplementary evaluation JSONs. |

## Design decisions

### Why edge-based structural metrics?

For line drawings, Canny edge IoU/F1 captures geometric layout better than pixel
or bounding-box overlap. Combined with perceptual (DISTS), semantic (CLIP), and
distributional (CMMD) metrics, the suite covers different failure modes.
