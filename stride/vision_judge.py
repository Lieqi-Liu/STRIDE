"""Vision LLM judge for open-ended FRQ answers (optional heavy dependency)."""
from __future__ import annotations

import json
import re
import statistics
from typing import Any

from .bleurt import NUSCENES_OPEN_FRQ_IDS, WAYMO_OPEN_FRQ_IDS, collect_open_frq_rows
from .io import strip_thinking
from .visual import load_nuscenes_images, load_waymo_images

DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


def build_vision_judge_prompt(task: dict[str, Any], candidate_answer: str) -> str:
    question = str(task.get("question", "")).strip()
    gt = str(task.get("ground_truth", "")).strip()
    return (
        "You are an expert grader for autonomous-driving free-response questions.\n"
        "You are shown the same driving frames the model saw. Use the images as the primary "
        "evidence. The reference answer is a helpful guide, not an absolute script — reward "
        "answers that are visually correct even if wording differs.\n"
        "Assign an integer score from 0 to 10.\n"
        "Scoring rubric:\n"
        "10 = fully correct and complete given the visuals\n"
        "7-9 = mostly correct, minor omissions/errors\n"
        "4-6 = partially correct\n"
        "1-3 = mostly incorrect but related\n"
        "0 = irrelevant or completely wrong\n"
        "Respond with ONLY a JSON object: {\"score\": <int>, \"rationale\": \"<short>\"}.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {gt}\n"
        f"Candidate answer: {candidate_answer}\n"
    )


def parse_judge_score(raw_text: str) -> float | None:
    text = strip_thinking(raw_text)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "score" in payload:
            return float(payload["score"])
    except Exception:
        pass
    match = re.search(r"\b([0-9]|10)\b", text)
    if match:
        return float(match.group(1))
    return None


def score_vision_judge_tasks(
    tasks: list[dict[str, Any]],
    *,
    image_backend: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    question_ids: set[str] | None = None,
    tensor_parallel_size: int = 2,
    max_tokens: int = 192,
    temperature: float = 0.0,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Score open FRQs with a vision LLM judge via vLLM.

    Requires vLLM + a Qwen3-VL (or compatible) checkpoint and local image roots.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "Vision judge requires vLLM. Install vLLM and a Qwen3-VL checkpoint, "
            "or skip this metric and report MCQ / BLEURT / trajectory only."
        ) from exc

    if question_ids is None:
        question_ids = (
            set(WAYMO_OPEN_FRQ_IDS) if image_backend == "waymo" else set(NUSCENES_OPEN_FRQ_IDS)
        )
    rows, skip_counts = collect_open_frq_rows(tasks, question_ids)
    if not rows:
        return {"scored_count": 0, "skip_counts": dict(skip_counts), "mean": None}

    llm = LLM(
        model=judge_model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=16384,
        enforce_eager=True,
    )
    sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = []
        for row in batch:
            answer = strip_thinking(str(row.get("model_response", "")))
            if image_backend == "waymo":
                images = load_waymo_images(row)
            else:
                images = load_nuscenes_images(row)
            prompt_text = build_vision_judge_prompt(row, answer)
            # vLLM multimodal: pass PIL images alongside text when supported.
            prompts.append(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": images},
                }
            )
        outputs = llm.generate(prompts, sampling)
        for row, output in zip(batch, outputs):
            raw = output.outputs[0].text if output.outputs else ""
            score = parse_judge_score(raw)
            idx = int(row["_task_index"])
            if score is None:
                tasks[idx]["llm_judge_score"] = None
                tasks[idx]["llm_judge_raw"] = raw
                tasks[idx]["llm_judge_mode"] = "vision"
                continue
            tasks[idx]["llm_judge_score"] = float(score)
            tasks[idx]["llm_judge_raw"] = raw
            tasks[idx]["llm_judge_mode"] = "vision"
            scores.append(float(score))

    return {
        "scored_count": len(scores),
        "skip_counts": dict(skip_counts),
        "mean": statistics.mean(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
    }
