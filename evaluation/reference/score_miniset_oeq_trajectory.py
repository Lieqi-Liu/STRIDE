#!/usr/bin/env python3
"""Score TRJ-5 / TRJ-6 OEQ waypoint predictions with L2 / ADE / minFDE-style metrics."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_miniset_model_benchmark import (
    build_benchmark_summary,
    read_json,
    safe_slug,
    write_json,
    MODEL_SPECS,
)

TRAJECTORY_IDS = {"TRJ-5", "TRJ-6"}
THINK_END = "</" + "think>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score TRJ-5/TRJ-6 waypoint OEQs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--response-json",
        action="append",
        default=[],
        type=Path,
        help="Score this response JSON directly (name = stem without _responses).",
    )
    parser.add_argument("--skip-summary-refresh", action="store_true")
    return parser.parse_args()


def strip_thinking(text: str) -> str:
    if THINK_END in text:
        return text.rsplit(THINK_END, 1)[-1].strip()
    return text.strip()


def _strip_markdown_fence(text: str) -> str:
    """Remove ``` / ```json wrappers if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _balanced_arrays(text: str) -> list[str]:
    """Return JSON array substrings with balanced brackets (outermost first)."""
    out: list[str] = []
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start : i + 1])
                start = None
    return out


def parse_points(text: str, expected: int | None = None) -> list[list[float]] | None:
    """Extract a JSON list of [x, y] points from model output."""
    cleaned = _strip_markdown_fence(strip_thinking(text))
    # Prefer balanced outer arrays (handles [[x,y], ...] and markdown fences).
    candidates: list[str] = _balanced_arrays(cleaned)
    # Fallback: non-greedy slices + whole text.
    for match in re.finditer(r"\[[\s\S]*?\]", cleaned):
        candidates.append(match.group(0))
    candidates.append(cleaned)

    parsed: list[list[float]] | None = None
    for cand in reversed(candidates):  # prefer later arrays (final answer)
        try:
            payload = json.loads(cand)
        except Exception:
            # Tolerate trailing commas / single quotes lightly.
            try:
                payload = json.loads(cand.replace("'", '"'))
            except Exception:
                continue
        if not isinstance(payload, list) or not payload:
            continue
        points: list[list[float]] = []
        ok = True
        for item in payload:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                ok = False
                break
            try:
                points.append([float(item[0]), float(item[1])])
            except Exception:
                ok = False
                break
        if not ok or not points:
            continue
        if expected is not None and len(points) != expected:
            # Keep as fallback if exact length not found later.
            if parsed is None:
                parsed = points
            continue
        return points
    return parsed


def expected_point_count(question_id: str) -> int:
    return 5 if question_id == "TRJ-5" else 4


def point_l2(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def trajectory_metrics(
    pred: list[list[float]],
    gt: list[list[float]],
) -> dict[str, float]:
    n = min(len(pred), len(gt))
    if n <= 0:
        raise ValueError("empty trajectory")
    pred_n = pred[:n]
    gt_n = gt[:n]
    step_errors = [point_l2(p, g) for p, g in zip(pred_n, gt_n)]
    # ADE: mean L2 over aligned timesteps
    ade = statistics.mean(step_errors)
    # FDE: L2 at final aligned timestep
    fde = step_errors[-1]
    # "mini FDE": min L2 between predicted final point and any GT point
    # (useful when timing is off but endpoint region is right)
    mini_fde = min(point_l2(pred_n[-1], g) for g in gt_n)
    mean_l2 = ade
    return {
        "traj_num_points_compared": float(n),
        "traj_ade": ade,
        "traj_fde": fde,
        "traj_mini_fde": mini_fde,
        "traj_mean_l2": mean_l2,
        "traj_final_l2": fde,
    }


def score_file(response_path: Path) -> dict[str, Any]:
    payload = read_json(response_path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"{response_path} must contain tasks list")

    scored_at = datetime.now().isoformat(timespec="seconds")
    by_id: dict[str, list[dict[str, float]]] = defaultdict(list)
    parse_fail = 0
    scored = 0

    for task in tasks:
        qid = str(task.get("id", ""))
        if qid not in TRAJECTORY_IDS:
            continue
        if str(task.get("question_format", "")).upper() != "OEQ":
            continue
        gt_raw = str(task.get("ground_truth", "")).strip()
        pred_raw = str(task.get("model_response", "")).strip()
        expected = expected_point_count(qid)
        gt = parse_points(gt_raw, expected=expected)
        pred = parse_points(pred_raw, expected=expected)
        # Clear prior metrics
        for key in (
            "traj_ade",
            "traj_fde",
            "traj_mini_fde",
            "traj_mean_l2",
            "traj_final_l2",
            "traj_num_points_compared",
            "traj_parse_ok",
            "traj_scored_at",
            "traj_expected_points",
            "traj_pred_points",
            "traj_gt_points",
        ):
            task.pop(key, None)

        task["traj_expected_points"] = expected
        task["traj_scored_at"] = scored_at
        task.pop("traj_invalid_format", None)
        if gt is None:
            task["traj_parse_ok"] = False
            parse_fail += 1
            continue
        if pred is None:
            # Same rule as unparsed MCQ: keep the sample, assign a failed score.
            # Invalid / missing waypoints are scored as a stay-at-origin trajectory.
            pred = [[0.0, 0.0] for _ in range(len(gt))]
            task["traj_parse_ok"] = False
            task["traj_invalid_format"] = True
            parse_fail += 1
        else:
            task["traj_parse_ok"] = True
            scored += 1
        task["traj_pred_points"] = pred
        task["traj_gt_points"] = gt
        metrics = trajectory_metrics(pred, gt)
        task.update(metrics)
        by_id[qid].append(metrics)

    def bucket(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "min": None, "max": None}
        ordered = sorted(values)
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "min": ordered[0],
            "max": ordered[-1],
        }

    per_id = {}
    overall_vals: dict[str, list[float]] = defaultdict(list)
    for qid, rows in sorted(by_id.items()):
        entry = {
            "ade": bucket([r["traj_ade"] for r in rows]),
            "fde": bucket([r["traj_fde"] for r in rows]),
            "mini_fde": bucket([r["traj_mini_fde"] for r in rows]),
            "mean_l2": bucket([r["traj_mean_l2"] for r in rows]),
        }
        per_id[qid] = entry
        for key in ("traj_ade", "traj_fde", "traj_mini_fde", "traj_mean_l2"):
            overall_vals[key].extend(r[key] for r in rows)

    summary = {
        "response_path": str(response_path.resolve()),
        "scored_count": scored,
        "parse_fail_count": parse_fail,
        "per_question_id": per_id,
        "overall": {
            "ade": bucket(overall_vals["traj_ade"]),
            "fde": bucket(overall_vals["traj_fde"]),
            "mini_fde": bucket(overall_vals["traj_mini_fde"]),
            "mean_l2": bucket(overall_vals["traj_mean_l2"]),
        },
        "scored_at": scored_at,
    }
    payload.setdefault("meta", {})
    payload["meta"]["oeq_trajectory_summary"] = summary
    write_json(response_path, payload)
    return summary


def main() -> None:
    args = parse_args()
    selected = set(args.model)
    summaries = []
    to_score: list[tuple[str, Path]] = []
    for spec in MODEL_SPECS:
        name = str(spec["name"])
        if selected and name not in selected:
            continue
        if args.response_json and not selected:
            # When only explicit response paths are given, skip MODEL_SPECS.
            continue
        path = args.output_dir / f"{safe_slug(name)}_responses.json"
        to_score.append((name, path))
    for path in args.response_json:
        stem = path.stem
        name = stem[: -len("_responses")] if stem.endswith("_responses") else stem
        to_score.append((name, path))

    for name, path in to_score:
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        print(f"[score] {name} -> {path}")
        summary = score_file(path)
        summaries.append({"name": name, **summary})
        overall = summary["overall"]
        print(
            f"  scored={summary['scored_count']} parse_fail={summary['parse_fail_count']} "
            f"ADE={overall['ade']['mean']} FDE={overall['fde']['mean']} "
            f"miniFDE={overall['mini_fde']['mean']}"
        )

    if not summaries:
        raise SystemExit("No trajectory response files scored.")

    if not args.skip_summary_refresh:
        summary_path = args.output_dir / "benchmark_summary.json"
        existing_by_name: dict[str, dict[str, Any]] = {}
        if summary_path.exists():
            prior = read_json(summary_path)
            for result in prior.get("results", []):
                if isinstance(result, dict) and result.get("name"):
                    existing_by_name[str(result["name"])] = result
        for item in summaries:
            name = item["name"]
            existing_by_name.setdefault(name, {"name": name})
            existing_by_name[name]["trajectory_summary"] = {
                k: item[k]
                for k in ("scored_count", "parse_fail_count", "overall", "per_question_id", "scored_at")
            }
        # Expert-only runs may not match VLM MODEL_SPECS; still write a compact summary.
        try:
            benchmark_summary = build_benchmark_summary(
                input_json=args.input_json,
                output_dir=args.output_dir,
                model_specs=[spec for spec in MODEL_SPECS if spec.get("enabled", True)],
                existing_results_by_name=existing_by_name,
            )
            for result in benchmark_summary.get("results", []):
                name = str(result.get("name", ""))
                if name in existing_by_name and "trajectory_summary" in existing_by_name[name]:
                    result["trajectory_summary"] = existing_by_name[name]["trajectory_summary"]
            # Attach expert names absent from MODEL_SPECS.
            present = {str(r.get("name")) for r in benchmark_summary.get("results", [])}
            for name, item in existing_by_name.items():
                if name not in present and "trajectory_summary" in item:
                    benchmark_summary.setdefault("results", []).append(
                        {"name": name, "trajectory_summary": item["trajectory_summary"]}
                    )
        except Exception as exc:
            benchmark_summary = {
                "input_json": str(args.input_json),
                "output_dir": str(args.output_dir),
                "results": [
                    {"name": s["name"], "trajectory_summary": {
                        k: s[k]
                        for k in ("scored_count", "parse_fail_count", "overall", "per_question_id", "scored_at")
                    }}
                    for s in summaries
                ],
                "note": f"compact expert summary ({exc})",
            }
        write_json(summary_path, benchmark_summary)
        print(f"[summary] refreshed {summary_path}")


if __name__ == "__main__":
    main()
