#!/usr/bin/env python3
"""STRIDE evaluation CLI.

Examples
--------
Score MCQ + trajectory from a prediction JSON (no GPU needed)::

    python -m stride.cli score \\
        --split nuscenes_v6 \\
        --predictions path/to/model_responses.json \\
        --metrics mcq,trajectory \\
        --output-dir runs/my_model

Also run BLEURT (downloads Elron/bleurt-base-512 on first use)::

    python -m stride.cli score \\
        --split waymo_v1 \\
        --predictions path/to/model_responses.json \\
        --metrics mcq,trajectory,bleurt \\
        --output-dir runs/my_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .aggregate import SPLIT_PATHS, aggregate_scores, index_predictions, load_tasks
from .bleurt import default_open_oeq_ids, score_bleurt_tasks
from .io import read_json, write_json
from .mcq import score_mcq_tasks
from .trajectory import score_trajectory_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STRIDE benchmark evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score a prediction JSON against a STRIDE split")
    score.add_argument(
        "--split",
        default="nuscenes_v6",
        help=f"Split name ({', '.join(SPLIT_PATHS)}) or path to GT JSON",
    )
    score.add_argument("--predictions", type=Path, required=True, help="Model responses JSON")
    score.add_argument(
        "--metrics",
        default="mcq,trajectory",
        help="Comma-separated: mcq,trajectory,bleurt,aggregate (vision judge is separate)",
    )
    score.add_argument("--output-dir", type=Path, default=Path("runs/stride_eval"))
    score.add_argument("--model-name", default=None, help="Display name (default: prediction stem)")
    score.add_argument("--bleurt-model", default="Elron/bleurt-base-512")
    score.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument(
        "--write-scored-predictions",
        action="store_true",
        help="Write predictions JSON with per-task scores filled in",
    )
    score.add_argument(
        "--score-answered-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only score tasks with a non-empty model_response (default: true). "
        "Use --no-score-answered-only to treat missing answers as incorrect over the full split.",
    )

    info = sub.add_parser("info", help="Print split statistics")
    info.add_argument("--split", default="nuscenes_v6")

    return parser.parse_args()


def cmd_info(args: argparse.Namespace) -> None:
    payload, tasks = load_tasks(args.split)
    meta = payload.get("meta", {})
    print(json.dumps({"split": args.split, "meta": meta, "n_tasks_loaded": len(tasks)}, indent=2))


def cmd_score(args: argparse.Namespace) -> None:
    metrics = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}
    payload, gt_tasks = load_tasks(args.split)
    pred_payload = read_json(args.predictions)
    pred_tasks = pred_payload.get("tasks", pred_payload if isinstance(pred_payload, list) else [])
    if not isinstance(pred_tasks, list):
        raise SystemExit("Predictions JSON must contain a top-level 'tasks' list")

    merged = index_predictions(gt_tasks, pred_tasks)
    if args.score_answered_only:
        tasks = [t for t in merged if str(t.get("model_response", "")).strip()]
        if not tasks:
            raise SystemExit("No tasks with non-empty model_response found in predictions")
    else:
        tasks = merged

    model_name = args.model_name or args.predictions.stem.replace("_responses", "")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "benchmark": "STRIDE",
        "split": str(args.split),
        "model": model_name,
        "predictions": str(args.predictions.resolve()),
        "n_tasks_gt": len(gt_tasks),
        "n_tasks_scored": len(tasks),
        "score_answered_only": bool(args.score_answered_only),
        "metrics": {},
    }

    if "mcq" in metrics or "aggregate" in metrics:
        mcq = score_mcq_tasks(tasks)
        report["metrics"]["mcq"] = mcq
        print(
            f"[mcq] n={mcq['n_mcq']} accuracy={mcq['accuracy']} "
            f"baseline={mcq['random_guess_baseline']}"
        )

    if "trajectory" in metrics or "aggregate" in metrics:
        traj = score_trajectory_tasks(tasks)
        report["metrics"]["trajectory"] = traj
        overall = traj["overall"]
        print(
            f"[traj] scored={traj['scored_count']} parse_fail={traj['parse_fail_count']} "
            f"ADE={overall['ade']['mean']} miniFDE={overall['mini_fde']['mean']}"
        )

    if "bleurt" in metrics:
        split_key = str(args.split)
        qids = default_open_oeq_ids(split_key)
        bleurt = score_bleurt_tasks(
            tasks,
            question_ids=qids,
            model_name=args.bleurt_model,
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=False,
        )
        report["metrics"]["bleurt"] = bleurt
        print(f"[bleurt] scored={bleurt['scored_count']} mean={bleurt['mean']}")

    if "aggregate" in metrics:
        report["metrics"]["aggregate"] = aggregate_scores(tasks)

    write_json(out_dir / f"{model_name}_score_report.json", report)
    print(f"[wrote] {out_dir / f'{model_name}_score_report.json'}")

    if args.write_scored_predictions:
        scored = {
            "meta": {
                **(pred_payload.get("meta") if isinstance(pred_payload, dict) else {}),
                "stride_scored": True,
                "split": str(args.split),
                "model": model_name,
            },
            "tasks": tasks,
        }
        write_json(out_dir / f"{model_name}_responses_scored.json", scored)
        print(f"[wrote] {out_dir / f'{model_name}_responses_scored.json'}")

    # Compact one-line leaderboard-style summary
    mcq = report["metrics"].get("mcq", {})
    traj = report["metrics"].get("trajectory", {})
    bleurt = report["metrics"].get("bleurt", {})
    summary = {
        "model": model_name,
        "mcq_accuracy": mcq.get("accuracy"),
        "open_oeq_bleurt_mean": bleurt.get("mean"),
        "traj_ade": (traj.get("overall") or {}).get("ade", {}).get("mean"),
        "traj_mini_fde": (traj.get("overall") or {}).get("mini_fde", {}).get("mean"),
    }
    write_json(out_dir / f"{model_name}_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "score":
        cmd_score(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
