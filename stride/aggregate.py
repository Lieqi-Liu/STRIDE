"""Dataset loading helpers and score aggregation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .io import read_json
from .mcq import score_mcq_tasks
from .trajectory import score_trajectory_tasks

REPO_ROOT = Path(__file__).resolve().parents[1]

SPLIT_PATHS = {
    "nuscenes_v6": REPO_ROOT / "STRIDE" / "nuScenes" / "questions.json",
    "waymo_v1": REPO_ROOT / "STRIDE" / "Waymo" / "questions.json",
    "nuscenes_mini": REPO_ROOT / "STRIDE" / "Mini" / "nuscenes_mini.json",
    "waymo_mini": REPO_ROOT / "STRIDE" / "Mini" / "waymo_mini.json",
}


def load_tasks(split_or_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a STRIDE split by name or an arbitrary JSON path."""
    key = str(split_or_path)
    path = SPLIT_PATHS.get(key, Path(split_or_path))
    payload = read_json(path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"{path} must contain a top-level tasks list")
    return payload, tasks


def index_predictions(
    gt_tasks: list[dict[str, Any]],
    pred_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge model_response fields from predictions onto a GT task list copy."""

    def key(task: dict[str, Any]) -> tuple[Any, ...]:
        return (
            task.get("id"),
            task.get("scene_id"),
            task.get("group_id"),
            task.get("object_id"),
            task.get("question_id"),
            task.get("question"),
        )

    pred_by_key = {key(t): t for t in pred_tasks}
    merged: list[dict[str, Any]] = []
    for gt in gt_tasks:
        row = dict(gt)
        pred = pred_by_key.get(key(gt))
        if pred is None and gt.get("question_id"):
            # Fallback: match by question_id alone when present (Waymo).
            for candidate in pred_tasks:
                if candidate.get("question_id") == gt.get("question_id"):
                    pred = candidate
                    break
        if pred is not None:
            for field in (
                "model_response",
                "model_predicted_option",
                "model_is_correct",
                "model_random_baseline",
                "bleurt_model_gt",
                "llm_judge_score",
                "llm_judge_mode",
                "traj_ade",
                "traj_fde",
                "traj_mini_fde",
                "traj_parse_ok",
            ):
                if field in pred:
                    row[field] = pred[field]
            if "model_response" not in row and "response" in pred:
                row["model_response"] = pred["response"]
        merged.append(row)
    return merged


def aggregate_scores(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate MCQ / already-written BLEURT / judge / traj fields on tasks."""
    mcq = score_mcq_tasks(tasks)

    bleurt_vals = [
        float(t["bleurt_model_gt"])
        for t in tasks
        if t.get("bleurt_model_gt") is not None
        and str(t.get("question_format", "")).upper() == "FRQ"
        and str(t.get("id", "")) not in {"TRJ-5", "TRJ-6"}
    ]
    judge_vals = [
        float(t["llm_judge_score"])
        for t in tasks
        if t.get("llm_judge_score") is not None
        and str(t.get("question_format", "")).upper() == "FRQ"
        and str(t.get("id", "")) not in {"TRJ-5", "TRJ-6"}
    ]
    traj = score_trajectory_tasks(tasks)

    def _mean(vals: list[float]) -> float | None:
        return (sum(vals) / len(vals)) if vals else None

    return {
        "n_tasks": len(tasks),
        "mcq": mcq,
        "open_frq_bleurt_mean": _mean(bleurt_vals),
        "open_frq_llm_judge_mean": _mean(judge_vals),
        "trajectory": traj,
    }


def write_leaderboard_row(path: Path, row: dict[str, Any], *, fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or [
        "model",
        "model_type",
        "n_tasks",
        "n_mcq",
        "n_frq",
        "mcq_accuracy",
        "mcq_random_guess_baseline",
        "open_frq_bleurt_mean",
        "open_frq_llm_judge_mean",
        "traj_ade",
        "traj_mini_fde",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
