#!/usr/bin/env python3
"""Build a blank prediction template from a STRIDE split.

Usage:
    python evaluation/make_prediction_template.py --split nuscenes_v6 --output preds/template.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stride.aggregate import load_tasks
from stride.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="nuscenes_v6")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    args = parser.parse_args()

    payload, tasks = load_tasks(args.split)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    out_tasks = []
    for t in tasks:
        row = {
            "id": t.get("id"),
            "question_id": t.get("question_id"),
            "task": t.get("task"),
            "question_format": t.get("question_format"),
            "scene_id": t.get("scene_id"),
            "group_id": t.get("group_id"),
            "object_id": t.get("object_id"),
            "question": t.get("question"),
            "choices": t.get("choices"),
            "model_response": "",
        }
        out_tasks.append(row)

    write_json(
        args.output,
        {
            "meta": {
                "benchmark": "STRIDE",
                "split": str(args.split),
                "n_tasks": len(out_tasks),
                "note": "Fill model_response for each task, then run: python -m stride.cli score ...",
            },
            "tasks": out_tasks,
        },
    )
    print(f"Wrote template with {len(out_tasks)} tasks -> {args.output}")


if __name__ == "__main__":
    main()
