"""MCQ answer parsing and accuracy scoring."""
from __future__ import annotations

import json
import re
from typing import Any

from .io import strip_thinking


def extract_answer_key(raw_text: str, choices: dict[str, Any] | None = None) -> str | None:
    """Parse a single option letter (A/B/C/...) from free-form model output."""
    text = strip_thinking(raw_text or "")
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            answer = payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip().upper()[:1]
    except Exception:
        pass

    normalized = text.strip().upper()
    if len(normalized) == 1 and normalized.isalpha():
        return normalized
    match = re.search(r"\b(?:ANSWER|OPTION)?\s*[:\-]?\s*\(?([A-Z])\)?\b", normalized)
    if match:
        return match.group(1)
    trailing = re.search(r"(?:^|\n)\s*\(?([A-Z])\)?\s*$", normalized)
    if trailing:
        return trailing.group(1)
    if choices:
        lowered = text.lower()
        for key, value in choices.items():
            if isinstance(value, str) and value.strip() and value.lower() in lowered:
                return str(key).upper()
    return None


def score_mcq_task(task: dict[str, Any]) -> dict[str, Any]:
    """Score one MCQ task in-place and return a compact result dict."""
    choices = task.get("choices") if isinstance(task.get("choices"), dict) else {}
    n_choices = max(len(choices), 1)
    baseline = 1.0 / n_choices
    predicted = extract_answer_key(str(task.get("model_response", "")), choices)
    gt = str(task.get("ground_truth", "")).strip().upper()[:1]
    correct = bool(predicted) and predicted == gt
    task["model_predicted_option"] = predicted
    task["model_is_correct"] = correct
    task["model_random_baseline"] = baseline
    return {
        "id": task.get("id"),
        "predicted": predicted,
        "ground_truth": gt,
        "correct": correct,
        "random_baseline": baseline,
    }


def score_mcq_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Score all MCQ tasks. Unparsed answers count as incorrect (not dropped)."""
    results = []
    for task in tasks:
        if str(task.get("question_format", "")).upper() != "MCQ":
            continue
        results.append(score_mcq_task(task))
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    baselines = [r["random_baseline"] for r in results]
    accuracy = (n_correct / n) if n else None
    baseline = (sum(baselines) / n) if n else None
    return {
        "n_mcq": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "random_guess_baseline": baseline,
        "accuracy_minus_baseline": (accuracy - baseline) if accuracy is not None and baseline is not None else None,
    }
