"""
mathvdiagram.benchmark
~~~~~~~~~~~~~~~~~~~~~~~

The benchmark half of MathVDiagram: take the curated ``concise_prompts.csv``
produced by the curation pipeline and

  1. generate    -- ask LLMs / image models to draw each diagram, compile to PNG
  2. evaluate    -- score generated images vs. ground truth (DISTS, CLIP, Edge, CMMD)
  3. compare     -- cross-model leaderboard with significance testing
  4. report      -- per-model side-by-side HTML report
  5. paper_figures -- publication figures, LaTeX tables, and data exports
  6. curate      -- local web UI to exclude low-quality prompts
  7. hf_upload   -- build and push the HuggingFace dataset

Submodules pull in heavy optional dependencies (torch, transformers, cairosvg,
matplotlib, ...), so they are not imported eagerly. Install them with::

    pip install "mathvdiagram[benchmark]"
"""

from dotenv import load_dotenv

# Ensure .env is loaded when any benchmark entry point runs.
load_dotenv()

__all__ = [
    "generate",
    "evaluate",
    "compare",
    "report",
    "paper_figures",
    "curate",
    "hf_upload",
    "models",
]
