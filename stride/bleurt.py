"""BLEURT scoring for open-ended OEQ text answers."""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .io import strip_thinking
from .trajectory import TRAJECTORY_IDS

# Default open-OEQ sets (geometric TRJ-5/6 are scored separately).
NUSCENES_OPEN_OEQ_IDS = {"SC-6", "SP-7", "SU-7", "TE-6", "TRJ-7"}
WAYMO_OPEN_OEQ_IDS = {"SC-4", "SP-3a", "TRJ-7", "TRJ-9"}
DEFAULT_BLEURT_MODEL = "Elron/bleurt-base-512"


def default_open_oeq_ids(split: str) -> set[str]:
    if split.startswith("waymo"):
        return set(WAYMO_OPEN_OEQ_IDS)
    return set(NUSCENES_OPEN_OEQ_IDS)


def collect_open_oeq_rows(
    tasks: list[dict[str, Any]],
    question_ids: set[str] | None = None,
    *,
    include_errors: bool = False,
    exclude_trajectory: bool = True,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for index, task in enumerate(tasks):
        if str(task.get("question_format", "")).upper() != "OEQ":
            counters["non_oeq"] += 1
            continue
        qid = str(task.get("id", ""))
        if exclude_trajectory and qid in TRAJECTORY_IDS:
            counters["trajectory_oeq"] += 1
            continue
        if question_ids is not None and qid not in question_ids:
            counters["filtered_task_id"] += 1
            continue
        gt = str(task.get("ground_truth", "")).strip()
        response = str(task.get("model_response", "")).strip()
        if not gt:
            counters["missing_ground_truth"] += 1
            continue
        if not response:
            counters["missing_model_response"] += 1
            continue
        if response.startswith("[ERROR]") and not include_errors:
            counters["error_model_response"] += 1
            continue
        row = dict(task)
        row["_task_index"] = index
        rows.append(row)
        counters["scored_candidates"] += 1
    return rows, counters


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def bleurt_scores_batch(
    hypotheses: list[str],
    references: list[str],
    *,
    model_name: str = DEFAULT_BLEURT_MODEL,
    hf_home: Path | None = None,
    local_files_only: bool = False,
    batch_size: int = 32,
    device: str = "auto",
) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_device(device)
    cache_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if hf_home is not None:
        cache_kwargs["cache_dir"] = str(Path(hf_home) / "hub")

    tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **cache_kwargs)
    model.to(device)
    model.eval()

    scores: list[float] = []
    for start in range(0, len(hypotheses), batch_size):
        batch_h = hypotheses[start : start + batch_size]
        batch_r = references[start : start + batch_size]
        encoded = tokenizer(
            batch_r,
            batch_h,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
        if logits.ndim == 0:
            scores.append(float(logits.detach().cpu().item()))
        else:
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
    return scores


def score_bleurt_tasks(
    tasks: list[dict[str, Any]],
    *,
    question_ids: set[str] | None = None,
    model_name: str = DEFAULT_BLEURT_MODEL,
    hf_home: Path | None = None,
    local_files_only: bool = False,
    batch_size: int = 32,
    device: str = "auto",
    include_errors: bool = False,
) -> dict[str, Any]:
    rows, skip_counts = collect_open_oeq_rows(
        tasks, question_ids, include_errors=include_errors
    )
    if not rows:
        return {
            "scored_count": 0,
            "skip_counts": dict(skip_counts),
            "mean": None,
            "min": None,
            "max": None,
        }

    hypotheses = [strip_thinking(str(row.get("model_response", ""))) for row in rows]
    references = [str(row.get("ground_truth", "")).strip() for row in rows]
    scores = bleurt_scores_batch(
        hypotheses,
        references,
        model_name=model_name,
        hf_home=hf_home,
        local_files_only=local_files_only,
        batch_size=batch_size,
        device=device,
    )

    for row, score in zip(rows, scores):
        idx = int(row["_task_index"])
        tasks[idx]["bleurt_model_gt"] = float(score)
        tasks[idx]["bleurt_model"] = model_name

    ordered = sorted(scores)
    return {
        "scored_count": len(scores),
        "skip_counts": dict(skip_counts),
        "mean": statistics.mean(scores),
        "min": ordered[0],
        "max": ordered[-1],
    }
