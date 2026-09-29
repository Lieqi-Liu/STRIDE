#!/usr/bin/env python3
"""Refresh Waymo benchmark_summary.json from response files (does not mix nuScenes specs)."""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_waymo_miniset_model_benchmark import MODEL_SPECS, nusc_bench


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "questions_with_answers_waymo_miniset_50_per_id_v1.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "miniset_model_benchmark_outputs"
OPEN_FRQ_IDS = {"SC-4", "SP-3a", "TRJ-7", "TRJ-9"}
TRAJECTORY_IDS = {"TRJ-5", "TRJ-6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


PROPRIETARY_DISPLAY = {
    "gpt-5.5": "GPT-5.5",
    "gemini-3.6-flash": "gemini-3.6-flash",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "claude-sonnet-5": "claude-sonnet-5",
}

EXPERT_DISPLAY = {
    "dolphins_gray311": "Dolphins",
}


def expert_results(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("expert_*/*_responses.json")):
        if "smoke" in path.parts:
            continue
        payload = nusc_bench.read_json(path)
        tasks = payload.get("tasks") or []
        if not tasks:
            continue
        stem = path.stem.removesuffix("_responses")
        display = EXPERT_DISPLAY.get(stem, stem)
        rows.append(
            {
                "name": display,
                "model_id": (payload.get("meta") or {}).get("expert_model") or stem,
                "backend": (payload.get("meta") or {}).get("backend") or "expert",
                "model_type": "expert",
                "status": "completed",
                "output_json": str(path),
                "summary": nusc_bench.summarize_model(tasks),
            }
        )
    return rows


def proprietary_results(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("proprietary_*/*_responses.json")):
        if "smoke" in path.parts:
            continue
        payload = nusc_bench.read_json(path)
        tasks = payload.get("tasks") or []
        if not tasks:
            continue
        stem = path.stem.removesuffix("_responses")
        rows.append(
            {
                "name": PROPRIETARY_DISPLAY.get(stem, stem),
                "model_id": (payload.get("meta") or {}).get("expert_model") or stem,
                "backend": (payload.get("meta") or {}).get("backend") or "proprietary",
                "model_type": "proprietary",
                "status": "completed",
                "output_json": str(path),
                "summary": nusc_bench.summarize_model(tasks),
            }
        )
    return rows


def leaderboard_row(name: str, result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary") or {}
    mcq = summary.get("mcq") or {}
    tasks = []
    output_json = result.get("output_json")
    if output_json and Path(output_json).exists():
        payload = nusc_bench.read_json(Path(output_json))
        tasks = payload.get("tasks") or []
    open_bleurt = [
        float(t["bleurt_model_gt"])
        for t in tasks
        if str(t.get("id")) in OPEN_FRQ_IDS and isinstance(t.get("bleurt_model_gt"), (int, float))
    ]
    open_judge = [
        float(t["llm_judge_score"])
        for t in tasks
        if str(t.get("id")) in OPEN_FRQ_IDS and isinstance(t.get("llm_judge_score"), (int, float))
    ]
    traj = [t for t in tasks if str(t.get("id")) in TRAJECTORY_IDS]
    traj_ok = [t for t in traj if t.get("traj_parse_ok") is True]
    # ADE/FDE include invalid formats (stay-at-origin penalty). Do not drop unparsed.
    traj_scored = [t for t in traj if isinstance(t.get("traj_ade"), (int, float))]
    judge_mode = next((str(t.get("llm_judge_mode")) for t in tasks if t.get("llm_judge_mode")), "")
    return {
        "model": name,
        "model_type": result.get("model_type") or "vlm",
        "split": "waymo_miniset_50_per_id_v1",
        "n_tasks": summary.get("total"),
        "n_mcq": (mcq.get("total")),
        "n_frq": (summary.get("frq") or {}).get("total"),
        "mcq_accuracy": mcq.get("accuracy"),
        "mcq_random_guess_baseline": mcq.get("random_guess_baseline"),
        "mcq_accuracy_minus_baseline": mcq.get("accuracy_minus_random_guess"),
        "open_frq_bleurt_mean": mean_or_none(open_bleurt),
        "open_frq_bleurt_n": len(open_bleurt),
        "open_frq_llm_judge_mean": mean_or_none(open_judge),
        "open_frq_llm_judge_n": len(open_judge),
        "llm_judge_mode": judge_mode,
        "traj_parse_ok": len(traj_ok),
        "traj_n": len(traj),
        "traj_ade": mean_or_none([float(t["traj_ade"]) for t in traj_scored]),
        "traj_fde": mean_or_none([float(t["traj_fde"]) for t in traj_scored]),
        "traj_mini_fde": mean_or_none(
            [float(t["traj_mini_fde"]) for t in traj_scored]
        ),
    }


def main() -> None:
    args = parse_args()
    original = nusc_bench.MODEL_SPECS
    nusc_bench.MODEL_SPECS = MODEL_SPECS
    try:
        enabled = [spec for spec in MODEL_SPECS if spec.get("enabled", True)]
        summary_path = args.output_dir / "benchmark_summary.json"
        existing_by_name: dict[str, dict[str, Any]] = {}
        last_run: list[dict[str, Any]] = []
        if summary_path.exists():
            prior = nusc_bench.read_json(summary_path)
            last_run = list(prior.get("last_run_model_specs") or [])
            for result in prior.get("results", []):
                name = str(result.get("name") or "")
                if name:
                    existing_by_name[name] = result
        benchmark_summary = nusc_bench.build_benchmark_summary(
            input_json=args.input_json,
            output_dir=args.output_dir,
            model_specs=enabled,
            last_run_model_specs=last_run,
            existing_results_by_name=existing_by_name,
        )
        extra = proprietary_results(args.output_dir) + expert_results(args.output_dir)
        existing_names = {str(r.get("name")) for r in benchmark_summary.get("results", [])}
        for result in extra:
            if result["name"] not in existing_names:
                benchmark_summary.setdefault("results", []).append(result)
        benchmark_summary["dataset_split"] = "waymo_miniset_50_per_id_v1"
        nusc_bench.write_json(summary_path, benchmark_summary)
        print(f"Wrote {summary_path}")

        rows = [leaderboard_row(str(r.get("name")), r) for r in benchmark_summary.get("results", [])]
        rows = [row for row in rows if row.get("n_tasks")]
        rows.sort(key=lambda r: (-(r.get("mcq_accuracy") or -1.0), str(r.get("model"))))
        if rows:
            csv_path = args.output_dir / "waymo_v1_leaderboard.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {csv_path}")
            for row in rows:
                print(
                    f"{row['model']}: mcq={row['mcq_accuracy']} "
                    f"bleurt={row['open_frq_bleurt_mean']} "
                    f"judge={row['open_frq_llm_judge_mean']} "
                    f"ade={row['traj_ade']} parse={row['traj_parse_ok']}/{row['traj_n']}"
                )
    finally:
        nusc_bench.MODEL_SPECS = original


if __name__ == "__main__":
    main()
