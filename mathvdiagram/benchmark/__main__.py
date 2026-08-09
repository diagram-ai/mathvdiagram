"""Allow ``python -m mathvdiagram.benchmark`` to print available subcommands."""

import sys

USAGE = """\
mathvdiagram.benchmark -- run individual stages with:

  python -m mathvdiagram.benchmark.generate       Generate diagrams from model suite
  python -m mathvdiagram.benchmark.evaluate        Score generated images vs ground truth
  python -m mathvdiagram.benchmark.compare         Cross-model leaderboard + significance
  python -m mathvdiagram.benchmark.report          Per-model side-by-side HTML report
  python -m mathvdiagram.benchmark.paper_figures   Publication figures, tables, exports
  python -m mathvdiagram.benchmark.curate          Local web UI for prompt curation
  python -m mathvdiagram.benchmark.hf_upload       Build + push HuggingFace dataset

Pass --help to any subcommand for options.
"""


def main() -> None:
    print(USAGE)
    sys.exit(0)


if __name__ == "__main__":
    main()
