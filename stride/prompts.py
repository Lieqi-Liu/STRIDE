"""Prompt builders shared by inference runners."""
from __future__ import annotations

from typing import Any

MASKED_OBJECT_REFERENCE = "the object in the red bounding box"


def build_text_prompt(task: dict[str, Any], *, num_images: int = 5) -> str:
    """Build the text portion of a STRIDE multimodal prompt."""
    question = str(task.get("question", "")).replace("<obj>", MASKED_OBJECT_REFERENCE)
    choices = task.get("choices", {})
    question_format = str(task.get("question_format", "MCQ"))
    qid = str(task.get("id", ""))

    if question_format.upper() == "MCQ" and isinstance(choices, dict) and choices:
        answer_instruction = (
            "Select exactly one option from the provided choices. "
            "Respond with only the option key, for example A."
        )
        choice_text = f"Choices: {choices}\n"
    elif qid in {"TRJ-5", "TRJ-6"}:
        n_pts = 5 if qid == "TRJ-5" else 4
        answer_instruction = (
            f"Predict {n_pts} future ego-centric waypoints as a JSON list of [x, y] pairs in meters, "
            f'for example [[0.0, 1.0], ...]. Respond with only the JSON list.'
        )
        choice_text = ""
    else:
        answer_instruction = "Provide a concise plain-text answer."
        choice_text = ""

    if qid == "SC-6":
        context_line = (
            f"You are given {num_images} chronological 360-degree multi-camera frames from one driving scene. "
            "Use the full clip to summarize the overall environment.\n"
        )
        target_line = ""
    else:
        context_line = (
            f"You are given {num_images} consecutive driving frames. The first {num_images - 1} images "
            f"are temporal context. The {num_images}th image is the query frame; when a target object "
            "exists, it is highlighted by a red bounding box.\n"
        )
        target_line = ""
        if task.get("object_id") or task.get("bbox_xyxy"):
            target_line = f"Target object: {MASKED_OBJECT_REFERENCE} in the {num_images}th image.\n"

    return (
        f"{context_line}"
        f"Question ID: {qid}\n"
        f"{target_line}"
        f"Question: {question}\n"
        f"{choice_text}"
        f"{answer_instruction}"
    )
