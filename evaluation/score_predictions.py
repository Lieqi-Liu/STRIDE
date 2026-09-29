#!/usr/bin/env python3
"""Score predictions against a STRIDE split.

    python evaluation/score_predictions.py \
        --split nuscenes_v6 \
        --predictions evaluation/examples/sample_predictions_nuscenes.json \
        --metrics mcq,trajectory \
        --output-dir eval_out/sample
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stride.cli import cmd_score, parse_args


def main() -> None:
    # Allow calling without the 'score' subcommand for convenience.
    if len(sys.argv) > 1 and sys.argv[1] != "score" and not sys.argv[1].startswith("-"):
        pass
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] != "score" and sys.argv[1].startswith("-")):
        sys.argv.insert(1, "score")
    args = parse_args()
    if args.command != "score":
        raise SystemExit("This script only supports the score command")
    cmd_score(args)


if __name__ == "__main__":
    main()
