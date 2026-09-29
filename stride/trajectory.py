"""Trajectory waypoint parsing and ADE / FDE / mini-FDE scoring for TRJ-5 / TRJ-6."""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from typing import Any

from .io import strip_thinking

TRAJECTORY_IDS = {"TRJ-5", "TRJ-6"}


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _balanced_arrays(text: str) -> list[str]:
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
    """Extract a JSON list of [x, y] waypoints from model output."""
    cleaned = _strip_markdown_fence(strip_thinking(text))
    candidates: list[str] = _balanced_arrays(cleaned)
    for match in re.finditer(r"\[[\s\S]*?\]", cleaned):
        candidates.append(match.group(0))
    candidates.append(cleaned)

    parsed: list[list[float]] | None = None
    for cand in reversed(candidates):
        try:
            payload = json.loads(cand)
        except Exception:
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
            if parsed is None:
                parsed = points
            continue
        return points
    return parsed


def expected_point_count(question_id: str) -> int:
    return 5 if question_id == "TRJ-5" else 4


def point_l2(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def trajectory_metrics(pred: list[list[float]], gt: list[list[float]]) -> dict[str, float]:
    n = min(len(pred), len(gt))
    if n <= 0:
        raise ValueError("empty trajectory")
    pred_n = pred[:n]
    gt_n = gt[:n]
    step_errors = [point_l2(p, g) for p, g in zip(pred_n, gt_n)]
    ade = statistics.mean(step_errors)
    fde = step_errors[-1]
    mini_fde = min(point_l2(pred_n[-1], g) for g in gt_n)
    return {
        "traj_num_points_compared": float(n),
        "traj_ade": ade,
        "traj_fde": fde,
        "traj_mini_fde": mini_fde,
        "traj_mean_l2": ade,
        "traj_final_l2": fde,
    }


def score_trajectory_task(task: dict[str, Any]) -> dict[str, Any] | None:
    qid = str(task.get("id", ""))
    if qid not in TRAJECTORY_IDS:
        return None
    if str(task.get("question_format", "")).upper() != "OEQ":
        return None

    expected = expected_point_count(qid)
    gt = parse_points(str(task.get("ground_truth", "")), expected=expected)
    pred = parse_points(str(task.get("model_response", "")), expected=expected)

    for key in (
        "traj_ade",
        "traj_fde",
        "traj_mini_fde",
        "traj_mean_l2",
        "traj_final_l2",
        "traj_num_points_compared",
        "traj_parse_ok",
        "traj_invalid_format",
        "traj_expected_points",
        "traj_pred_points",
        "traj_gt_points",
    ):
        task.pop(key, None)

    task["traj_expected_points"] = expected
    if gt is None:
        task["traj_parse_ok"] = False
        return {"id": qid, "parse_ok": False, "invalid_format": False}

    invalid = False
    if pred is None:
        # Keep the sample; score stay-at-origin as a failed prediction.
        pred = [[0.0, 0.0] for _ in range(len(gt))]
        task["traj_parse_ok"] = False
        task["traj_invalid_format"] = True
        invalid = True
    else:
        task["traj_parse_ok"] = True

    task["traj_pred_points"] = pred
    task["traj_gt_points"] = gt
    metrics = trajectory_metrics(pred, gt)
    task.update(metrics)
    return {"id": qid, "parse_ok": not invalid, "invalid_format": invalid, **metrics}


def _bucket(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def score_trajectory_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, list[dict[str, float]]] = defaultdict(list)
    parse_fail = 0
    scored = 0
    for task in tasks:
        result = score_trajectory_task(task)
        if result is None:
            continue
        if not result.get("parse_ok"):
            parse_fail += 1
        else:
            scored += 1
        if "traj_ade" in result:
            by_id[str(result["id"])].append(result)

    per_id = {}
    overall_vals: dict[str, list[float]] = defaultdict(list)
    for qid, rows in sorted(by_id.items()):
        entry = {
            "ade": _bucket([r["traj_ade"] for r in rows]),
            "fde": _bucket([r["traj_fde"] for r in rows]),
            "mini_fde": _bucket([r["traj_mini_fde"] for r in rows]),
        }
        per_id[qid] = entry
        for key in ("traj_ade", "traj_fde", "traj_mini_fde"):
            overall_vals[key].extend(r[key] for r in rows)

    return {
        "scored_count": scored,
        "parse_fail_count": parse_fail,
        "per_question_id": per_id,
        "overall": {
            "ade": _bucket(overall_vals["traj_ade"]),
            "fde": _bucket(overall_vals["traj_fde"]),
            "mini_fde": _bucket(overall_vals["traj_mini_fde"]),
        },
    }
