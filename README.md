# MathVDiagram

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/diagramAI/mathvision-diagram-benchmark)

A benchmark for evaluating LLMs and image models on **precision mathematical
diagram generation**: given a natural-language description of a math diagram,
can a model redraw it faithfully?

MathVDiagram is organized as two stages that share a single artifact, the
curated prompt set (`concise_prompts.csv`):

1. **Prompt curation** (`mathvdiagram`) — process the
   [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) dataset
   (3,040 images across 16 subjects): classify images into a diagram taxonomy,
   describe each with an ensemble of vision-language models, and aggregate the
   descriptions into one authoritative prompt with an open-source judge.
2. **Benchmark** (`mathvdiagram.benchmark`) — feed those prompts to a suite of
   models, compile their output to images, and score them against ground truth
   with four automated metrics; then produce a leaderboard, paper figures, and
   a HuggingFace dataset.

```
MathVision (3,040 images)
      │   Stage 1: curation  (classify → describe → aggregate → curate)
      ▼
data/concise_prompts.csv   ← the hand-off artifact
      │   Stage 2: benchmark (generate → evaluate → compare → paper → upload)
      ▼
Leaderboard + paper assets + HuggingFace dataset
```

---

## Stage 1: Prompt curation

### Why an ensemble + an independent judge

Benchmarking diagram generation needs precise descriptions — detailed enough
for a model to recreate a diagram exactly. A single VLM has blind spots: it may
capture geometric labels but miss line styles, or describe axes but overlook
angle marks.

**Description (Step 2)** uses an ensemble of proprietary VLMs with different
vision encoders, each independently answering the same 12-point checklist:

| Model |
|-------|
| OpenAI GPT-4o |
| Gemini 2.5 Flash |
| Claude Sonnet 4 |
| Llama 3.2 (optional, via Groq) |

**Aggregation (Step 3)** uses an **open-source** VLM judge (Qwen3-VL via
OpenRouter, or Llama via Groq). The judge sees the original image plus all
descriptions, then: identifies agreement, resolves conflicts against the image,
and adds anything every describer missed. Using an open-source judge keeps the
aggregation step reproducible and avoids circularity — the judge never grades
its own work.

### Diagram taxonomy

During classification, images are sorted into six categories, each with tailored
description hints:

| Category | Definition | Examples |
|----------|-----------|----------|
| **Geometric Construction** | Abstract 2D shapes with formal annotations | Labeled triangles, circle theorems |
| **Coordinate Plot** | Explicit coordinate systems and axes | Cartesian/polar plots, shaded regions |
| **Statistical Chart** | Quantitative data with labeled axes/categories | Bar, pie, histogram, scatter |
| **Schematic Diagram** | Structured relationships without axes | Venn diagrams, number lines, trees, graphs |
| **3D Figure** | Wireframe/technical drawings of 3D objects | Labeled prisms, cross-sections, nets |
| **Non-Diagram** | Real-world objects, photos, illustrations | Counting pictures, illustrated puzzles |

### Curation pipeline

```
Step 1: Classification (dataset_helper/)
  MathVision (3,040 images)
    → Metadata pre-filter (high-tier subjects → no API calls)
    → Structured LLM classification (mixed/low-tier images)
    → Boolean-based diagram / non-diagram decision
    → full_classification.csv

Step 2: Description (describe.py)
    → GPT-4o, Gemini 2.5 Flash, Claude Sonnet 4 (independent, same checklist)
    → descriptions.csv

Step 3: Aggregation (consensus.py)
    → open-source VLM judge (Qwen3-VL / Llama)
    → agreement analysis, conflict resolution, coverage check
    → aggregated_descriptions.csv / concise_prompts.csv

Step 4: Report (report.py)
    → HTML report with embedded images and descriptions
```

---

## Stage 2: Benchmark

Once you have a curated `data/concise_prompts.csv`, the
[`mathvdiagram.benchmark`](mathvdiagram/benchmark/README.md) subpackage runs the
evaluation:

| Step | Module | Output |
|------|--------|--------|
| Generate | `benchmark.generate` | Prompt each model; compile TikZ / SVG / Python (or take image output) to `outputs/<model>/*.png` |
| Evaluate | `benchmark.evaluate` | DISTS, CLIP similarity, Edge IoU, Edge F1, and CMMD → `outputs/<model>/eval_results.json` |
| Compare | `benchmark.compare` | Cross-model leaderboard + Wilcoxon/Holm significance → `outputs/comparison_report.html` |
| Report | `benchmark.report` | Per-model side-by-side HTML |
| Paper | `benchmark.paper_figures` | PDF figures, LaTeX tables, CSV/JSON exports into `paper/` |
| Curate | `benchmark.curate` | Local web UI to record `data/excluded_ids.txt` |
| Upload | `benchmark.hf_upload` | Build + push the HuggingFace dataset |

### Models

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

### Metrics

- **DISTS** (lower is better): perceptual distance from deep features.
- **CLIP Similarity** (higher is better): cosine similarity of CLIP ViT-B/32 embeddings.
- **Edge IoU** / **Edge F1** (higher is better): overlap of Canny edge masks; well suited to line drawings.
- **CMMD** (lower is better): CLIP Maximum Mean Discrepancy (unbiased MMD, median-heuristic bandwidth).

---

## Installation

```bash
git clone https://github.com/diagram-ai/mathvdiagram
cd mathvdiagram
python -m venv venv && source venv/bin/activate

# Stage 1 (curation) only:
pip install -e .

# Stage 1 + Stage 2 (benchmark: metrics, figures, HF upload):
pip install -e ".[benchmark]"
```

The benchmark generation step also needs a system TeX stack for the TikZ
compiler — a [TeX Live](https://www.tug.org/texlive) install providing
`pdflatex`, plus `pdftoppm` from [poppler](https://poppler.freedesktop.org) —
and Cairo for the SVG compiler. On a fresh VM, `examples/setup_gcp.sh`
installs all of this.

Then configure API keys:

```bash
cp .env.example .env   # fill in the keys for the providers you use
```

---

## Usage

All commands run from the repository root.

### Stage 1: build the prompt set

```bash
# Full curation pipeline
python -m mathvdiagram.pipeline

# Smoke test on a few images
python -m mathvdiagram.pipeline --num-samples 10

# Reuse completed steps
python -m mathvdiagram.pipeline --skip-classify --skip-describe

# Run classification independently (+ standalone report)
python -m mathvdiagram.dataset_helper --classify --report
```

Review the prompts, then place the curated file at `data/concise_prompts.csv`:

```bash
python -m mathvdiagram.benchmark.curate
```

### Stage 2: run the benchmark

```bash
# One model, a few prompts
python -m mathvdiagram.benchmark.generate --model deepseek-v3 --limit 20 --workers 4
python -m mathvdiagram.benchmark.evaluate --model deepseek-v3
python -m mathvdiagram.benchmark.report   --model deepseek-v3

# All models end-to-end, then compare + paper assets
bash examples/run_experiment.sh
python -m mathvdiagram.benchmark.compare
python -m mathvdiagram.benchmark.paper_figures

# Publish
python -m mathvdiagram.benchmark.hf_upload --dry-run
```

See [`examples/`](examples/) for the orchestration scripts and
[`mathvdiagram/benchmark/`](mathvdiagram/benchmark/README.md) for per-module docs.

---

## Repository structure

```
mathvdiagram/
├── pyproject.toml            # packaging + optional-dependency extras
├── README.md
├── .env.example
├── data/                     # concise_prompts.csv, excluded_ids.txt, ground_truth/ (gitignored)
├── paper/                    # generated figures, tables, data exports, reports
├── examples/                 # run_experiment.sh, setup_gcp.sh
├── tests/
└── mathvdiagram/
    ├── config.py             # keys, model names, prompts, output paths
    ├── api_clients.py        # client factories (OpenAI, Gemini, Claude, Qwen, Llama)
    ├── data_loader.py        # MathVision loading + image access
    ├── utils.py              # retry, base64, checkpoint I/O
    ├── classify.py           # legacy GPT binary classification (fallback)
    ├── describe.py           # Step 2: ensemble VLM descriptions
    ├── consensus.py          # Step 3: open-source judge aggregation
    ├── report.py             # Step 4: curation HTML report
    ├── pipeline.py           # curation orchestrator (CLI: python -m mathvdiagram.pipeline)
    ├── dataset_helper/       # Step 1: taxonomy classification subsystem
    └── benchmark/            # Stage 2: generate, evaluate, compare, report,
                              #          paper_figures, curate, hf_upload, models
```

---

## Datasets & models

The generated images, ground truth, and per-image metrics are published on the
Hugging Face Hub:
[`diagramAI/mathvision-diagram-benchmark`](https://huggingface.co/datasets/diagramAI/mathvision-diagram-benchmark),
with three configs — `full`, `common_subset`, and `ground_truth`. Rebuild and
push it with `python -m mathvdiagram.benchmark.hf_upload`.

Source images come from [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision)
(Wang et al., 2024).

---

## Design decisions

**Boolean-based classification over confidence thresholds.** LLM confidence is
poorly calibrated, so classification derives six concrete boolean features
(`has_geometric_labels`, `has_real_world_objects`, ...) and applies deterministic
rules, making decisions transparent and reproducible.

**Ensemble describers + one judge over a single model.** Different vision
encoders miss different details; three independent descriptions are
complementary, and the judge verifies against the image rather than voting.

**Metadata pre-filtering.** Subjects like analytic and solid geometry are almost
entirely formal diagrams, so they skip LLM classification, saving a large share
of API calls with negligible accuracy loss.

**Open-source aggregation judge.** Using a proprietary describer as the judge
would bias it toward its own output; an open-source judge keeps aggregation
reproducible and non-circular.

**Edge-based structural metrics.** For line drawings, Canny edge IoU/F1 captures
geometric layout better than pixel or bounding-box overlap.

---

## Citation

If you use MathVDiagram, please cite it. (Update with the final venue/DOI once
published.)

```bibtex
@software{mathvdiagram2026,
  title  = {MathVDiagram: A Benchmark for Precision Mathematical Diagram Generation},
  author = {Kashyap, Harish and CR, Sriram and Tuti, Sanyukta},
  year   = {2026},
  url    = {https://github.com/diagram-ai/mathvdiagram}
}
```

## Acknowledgments

Source images are from the [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision)
dataset. The repository organization is inspired by
[DeTikZify](https://github.com/potamides/DeTikZify).
