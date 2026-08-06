# MathVDiagram<br><sub><sup>A Benchmark for Precision Mathematical Diagram Generation</sup></sub>
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/diagramAI/mathvision-diagram-benchmark)

MathVDiagram evaluates how faithfully LLMs and image-generation models can
redraw mathematical diagrams from natural-language descriptions. It processes
the [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) dataset
(2,920 images across 16 math subjects) in two stages: a **curation pipeline**
that builds precise prompts from an ensemble of vision-language models, and a
**benchmark** that generates diagrams with a suite of models, compiles them,
and scores them against ground truth.

## News
* **2026-08-06**: Repository reorganized into a two-stage structure
  (curation + benchmark). Packaging migrated to `pyproject.toml` with optional
  extras. Benchmark subpackage added: generate, evaluate, compare, report,
  paper figures, curate, and HuggingFace upload.
* **2026-03-06**: Initial release.

## Installation

MathVDiagram can be installed using [pip](https://pip.pypa.io/en/stable):
```sh
git clone https://github.com/diagram-ai/mathvdiagram
cd mathvdiagram

# Stage 1 (curation) only:
pip install -e .

# Stage 1 + Stage 2 (benchmark):
pip install -e ".[benchmark]"
```
In addition, the benchmark generation step requires a full
[TeX Live](https://www.tug.org/texlive) installation providing `pdflatex`,
[poppler](https://poppler.freedesktop.org) for `pdftoppm`, and
[Cairo](https://cairographics.org) for SVG compilation. On a fresh VM,
[`examples/setup_gcp.sh`](examples/setup_gcp.sh) installs all of this.

Then configure API keys:
```sh
cp .env.example .env   # fill in the keys for the providers you use
```

## Usage

> [!TIP]
> For detailed CLI options, pipeline architecture, and per-file documentation,
> see the subpackage READMEs:
> [Stage 1 (curation)](mathvdiagram/README.md) and
> [Stage 2 (benchmark)](mathvdiagram/benchmark/README.md).

All commands run from the repository root.

### Stage 1: build the prompt set

```sh
# Full curation pipeline
python -m mathvdiagram.pipeline

# Smoke test on a few images
python -m mathvdiagram.pipeline --num-samples 10

# Run classification independently
python -m mathvdiagram.dataset_helper --classify --report
```

Review the prompts, then curate:
```sh
python -m mathvdiagram.benchmark.curate
```

### Stage 2: run the benchmark

```sh
# One model, a few prompts
python -m mathvdiagram.benchmark.generate --model deepseek-v3 --limit 20 --workers 4
python -m mathvdiagram.benchmark.evaluate --model deepseek-v3
python -m mathvdiagram.benchmark.report   --model deepseek-v3

# All models end-to-end, then compare + paper assets
bash examples/run_experiment.sh
python -m mathvdiagram.benchmark.compare
python -m mathvdiagram.benchmark.paper_figures

# Publish to HuggingFace
python -m mathvdiagram.benchmark.hf_upload --dry-run
```

More involved examples can be found in the [examples](examples/) folder.

## Model Weights & Datasets

The generated images, ground truth, and per-image metrics are published on the
Hugging Face Hub:
[`diagramAI/mathvision-diagram-benchmark`](https://huggingface.co/datasets/diagramAI/mathvision-diagram-benchmark),
with three configs -- `full`, `common_subset`, and `ground_truth`. Rebuild and
push with `python -m mathvdiagram.benchmark.hf_upload`.

Source images come from
[MathVision](https://huggingface.co/datasets/MathLLMs/MathVision)
(Wang et al., 2024).

## Citation

If MathVDiagram has been beneficial for your research or applications, please
cite it. (Update with the final venue/DOI once published.)

```bibtex
@software{mathvdiagram2026,
  title  = {MathVDiagram: A Benchmark for Precision Mathematical Diagram Generation},
  author = {Kashyap, Harish and CR, Sriram and Tuti, Sanyukta and Mistry, Aryan and Kiran},
  year   = {2026},
  url    = {https://github.com/diagram-ai/mathvdiagram}
}
```

## Acknowledgments

Source images are from the
[MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) dataset. The
repository organization is inspired by
[DeTikZify](https://github.com/potamides/DeTikZify).
