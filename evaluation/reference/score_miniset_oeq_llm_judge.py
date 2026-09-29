#!/usr/bin/env python3
"""LLM-as-judge scoring for open-ended OEQs using Qwen3-VL-32B-Instruct.

Default mode is vision+text: the judge sees the same driving frames as the model
and treats ground truth as a reference (not absolute), following human-scoring notes.
Use --text-only for the legacy text-vs-GT judge.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_miniset_model_benchmark import (
    MODEL_SPECS,
    REPO_ROOT,
    build_benchmark_summary,
    read_json,
    safe_slug,
    write_json,
)

# Geometric waypoint OEQs are scored separately; exclude from LLM judge.
OPEN_OEQ_IDS = {"SC-6", "SP-7", "SU-7", "TE-6", "TRJ-7", "TM-6"}
THINK_END = "</" + "think>"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_HF_HOME = Path("/local1/lieqiliu/huggingface")
DEFAULT_FORMATTED_SCENES_DIR = Path("/local1/lieqiliu/nuscenes/fullset/formatted_scenes")
ANNOTATOR_PATH = REPO_ROOT / "lieqiliu" / "run_fake_full_vlm_batch.py"


def load_annotator() -> Any:
    spec = importlib.util.spec_from_file_location("full_nuscenes_vlm_annotator", ANNOTATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load annotator from {ANNOTATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge for open OEQ responses.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], help="Candidate model name(s) to score.")
    parser.add_argument(
        "--response-json",
        action="append",
        default=[],
        type=Path,
        help="Score this response JSON directly (repeatable). Name = stem without _responses.",
    )
    parser.add_argument(
        "--image-backend",
        choices=("nuscenes", "waymo"),
        default="nuscenes",
        help="Vision-judge image loader. waymo uses CAM_FRONT paths on the task.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--formatted-scenes-dir", type=Path, default=DEFAULT_FORMATTED_SCENES_DIR)
    parser.add_argument("--nuscenes-root", type=Path, default=DEFAULT_FORMATTED_SCENES_DIR.parent)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=2, help="Lower default for vision batches.")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Vision judge needs more context than text-only; raise for more SC-6 frames.",
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--skip-summary-refresh", action="store_true")
    parser.add_argument(
        "--skip-existing-vision",
        action="store_true",
        help="Resume by skipping rows that already have a parsed vision-judge score.",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Legacy mode: compare candidate to GT text only (no images).",
    )
    parser.add_argument(
        "--sc6-max-frames",
        type=int,
        default=8,
        help="Evenly subsample SC-6 scene frames for the vision judge (40 is too heavy).",
    )
    parser.add_argument(
        "--only-question-id",
        action="append",
        default=[],
        help="Override the default open-OEQ id set. Repeatable.",
    )
    return parser.parse_args()


def strip_thinking(text: str) -> str:
    if THINK_END in text:
        return text.rsplit(THINK_END, 1)[-1].strip()
    return text.strip()


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def subsample_images(images: list[Any], max_frames: int) -> list[Any]:
    if max_frames <= 0 or len(images) <= max_frames:
        return images
    keep = evenly_spaced_indices(len(images), max_frames)
    return [images[i] for i in keep]


def build_judge_prompt_text(task: dict[str, Any], candidate_answer: str) -> str:
    """Legacy text-only judge prompt."""
    question = str(task.get("question", "")).strip()
    gt = str(task.get("ground_truth", "")).strip()
    return (
        "You are an expert grader for autonomous-driving free-response questions.\n"
        "Compare the candidate answer to the ground-truth answer and assign an integer score "
        "from 0 to 10.\n"
        "Scoring rubric:\n"
        "10 = fully correct and complete\n"
        "7-9 = mostly correct, minor omissions/errors\n"
        "4-6 = partially correct\n"
        "1-3 = mostly incorrect but related\n"
        "0 = irrelevant or completely wrong\n"
        "Respond with ONLY a JSON object of the form "
        '{"score": <int 0-10>, "rationale": "<one short sentence>"}.\n\n'
        f"Question ID: {task.get('id', '')}\n"
        f"Question: {question}\n"
        f"Ground truth: {gt}\n"
        f"Candidate answer: {candidate_answer}\n"
    )


def build_judge_prompt_vision(
    task: dict[str, Any],
    candidate_answer: str,
    *,
    num_images: int,
    is_sc6: bool,
) -> str:
    """Vision+text judge prompt informed by human scoring notes.

    Human notes emphasized: trust frames over GT when they conflict (weather,
    lighting, construction, object attributes); GT is a reference; penalize
    hallucinations not visible in frames; do not collapse scores for minor
    distance/localization errors when the rest is correct.
    """
    question = str(task.get("question", "")).strip()
    gt = str(task.get("ground_truth", "")).strip()
    obj_ref = str(task.get("object_reference") or "").strip()

    if is_sc6:
        frame_line = (
            f"You are given {num_images} chronological multi-camera scene frames from one driving clip.\n"
        )
    else:
        frame_line = (
            f"You are given {num_images} consecutive driving frames. "
            "The first images are temporal context; the last image is the query frame. "
            "When a target object exists, it is highlighted by a red bounding box in the query frame.\n"
        )

    target_line = ""
    if obj_ref:
        target_line = f"Target object reference (may be approximate): {obj_ref}\n"

    return (
        "You are an expert grader for autonomous-driving free-response questions.\n"
        f"{frame_line}"
        "Grade the candidate answer using BOTH the images and the reference ground-truth text.\n"
        "Priority rules (important):\n"
        "1) Visual evidence in the frames is primary. If weather/lighting/construction/object type/"
        "scene layout in the images disagree with the ground-truth text, trust the images.\n"
        "2) Treat ground truth as a helpful reference, not an absolute oracle.\n"
        "3) Penalize claims that are not supported by the images (hallucinations).\n"
        "4) Do not heavily penalize minor distance/localization numeric errors if the rest of the "
        "answer is visually correct.\n"
        "5) Object identity should follow the boxed/query object when present "
        "(e.g. bbox color is not the object color).\n"
        "Scoring rubric:\n"
        "10 = fully correct and complete vs the scene\n"
        "7-9 = mostly correct, minor omissions/errors\n"
        "4-6 = partially correct\n"
        "1-3 = mostly incorrect but related\n"
        "0 = irrelevant or completely wrong\n"
        "Respond with ONLY a JSON object of the form "
        '{"score": <int 0-10>, "rationale": "<one short sentence>"}.\n\n'
        f"Question ID: {task.get('id', '')}\n"
        f"{target_line}"
        f"Question: {question}\n"
        f"Reference ground truth: {gt}\n"
        f"Candidate answer: {candidate_answer}\n"
    )


def build_vision_request(
    processor: Any,
    task: dict[str, Any],
    candidate_answer: str,
    images: list[Any],
    *,
    is_sc6: bool,
) -> dict[str, Any]:
    prompt_text = build_judge_prompt_vision(
        task,
        candidate_answer,
        num_images=len(images),
        is_sc6=is_sc6,
    )
    image_content = [{"type": "image"} for _ in images]
    messages = [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": prompt_text}],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": images}}


def parse_judge_score(text: str) -> tuple[float | None, str]:
    cleaned = strip_thinking(text).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and "score" in payload:
            score = float(payload["score"])
            rationale = str(payload.get("rationale", "")).strip()
            return max(0.0, min(10.0, score)), rationale
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and "score" in payload:
                score = float(payload["score"])
                rationale = str(payload.get("rationale", "")).strip()
                return max(0.0, min(10.0, score)), rationale
        except Exception:
            pass
    match = re.search(r"\b([0-9]|10)(?:\.0+)?\b", cleaned)
    if match:
        return float(match.group(1)), cleaned[:240]
    return None, cleaned[:240]


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


def collect_targets(
    tasks: list[dict[str, Any]],
    max_rows: int,
    *,
    question_ids: set[str] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    allowed = question_ids if question_ids else OPEN_OEQ_IDS
    rows: list[tuple[int, dict[str, Any]]] = []
    for index, task in enumerate(tasks):
        if str(task.get("question_format", "")).upper() != "OEQ":
            continue
        qid = str(task.get("id", ""))
        if qid not in allowed:
            continue
        gt = str(task.get("ground_truth", "")).strip()
        response = str(task.get("model_response", "")).strip()
        if not gt or not response or response.startswith("[ERROR]"):
            continue
        rows.append((index, task))
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def preserve_text_judge_fields(task: dict[str, Any]) -> None:
    """Keep prior text-only judge scores when switching to vision."""
    mode = str(task.get("llm_judge_mode") or "")
    if mode == "vision":
        return
    if task.get("llm_judge_score") is None:
        return
    if "llm_judge_text_score" in task:
        return
    task["llm_judge_text_score"] = task.get("llm_judge_score")
    task["llm_judge_text_rationale"] = task.get("llm_judge_rationale")
    task["llm_judge_text_raw"] = task.get("llm_judge_raw")
    task["llm_judge_text_model"] = task.get("llm_judge_model")
    task["llm_judge_text_scored_at"] = task.get("llm_judge_scored_at")


def apply_judge_result(
    task: dict[str, Any],
    *,
    judge_model_id: str,
    mode: str,
    raw: str,
    score: float | None,
    rationale: str,
    scored_at: str,
    num_images: int | None = None,
    image_error: str | None = None,
) -> bool:
    """Write judge fields. Returns True if parse failed."""
    if mode == "vision":
        preserve_text_judge_fields(task)
    task["llm_judge_model"] = judge_model_id
    task["llm_judge_mode"] = mode
    task["llm_judge_raw"] = raw
    task["llm_judge_rationale"] = rationale
    task["llm_judge_scored_at"] = scored_at
    if num_images is not None:
        task["llm_judge_num_images"] = num_images
    if image_error:
        task["llm_judge_image_error"] = image_error
    elif "llm_judge_image_error" in task:
        task.pop("llm_judge_image_error", None)
    if score is None:
        task["llm_judge_score"] = None
        task["llm_judge_parse_ok"] = False
        return True
    task["llm_judge_score"] = score
    task["llm_judge_parse_ok"] = True
    return False


def score_response_file_text(
    response_path: Path,
    *,
    judge_llm: Any,
    sampling_params: Any,
    judge_model_id: str,
    batch_size: int,
    save_every: int,
    max_rows: int,
    question_ids: set[str] | None = None,
) -> dict[str, Any]:
    payload = read_json(response_path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"{response_path} must contain tasks")

    targets = collect_targets(tasks, max_rows, question_ids=question_ids)
    scored_at = datetime.now().isoformat(timespec="seconds")
    by_id: dict[str, list[float]] = defaultdict(list)
    parse_fail = 0
    completed_since_save = 0

    for offset in range(0, len(targets), batch_size):
        batch = targets[offset : offset + batch_size]
        prompts = []
        for _, task in batch:
            cand = strip_thinking(str(task.get("model_response", "")))
            prompts.append(build_judge_prompt_text(task, cand))
        outputs = judge_llm.generate(prompts, sampling_params=sampling_params)
        for (_, task), output in zip(batch, outputs):
            raw = output.outputs[0].text.strip() if output.outputs else ""
            score, rationale = parse_judge_score(raw)
            failed = apply_judge_result(
                task,
                judge_model_id=judge_model_id,
                mode="text",
                raw=raw,
                score=score,
                rationale=rationale,
                scored_at=scored_at,
            )
            if failed:
                parse_fail += 1
            elif score is not None:
                by_id[str(task.get("id", ""))].append(score)
            completed_since_save += 1

        if save_every > 0 and completed_since_save >= save_every:
            write_json(response_path, payload)
            completed_since_save = 0
            print(f"  checkpoint {min(offset + batch_size, len(targets))}/{len(targets)}")

    all_scores = [s for vals in by_id.values() for s in vals]
    summary = {
        "response_path": str(response_path.resolve()),
        "judge_model": judge_model_id,
        "judge_mode": "text",
        "candidate_count": len(targets),
        "scored_count": len(all_scores),
        "parse_fail_count": parse_fail,
        "overall": bucket(all_scores),
        "per_question_id": {qid: bucket(vals) for qid, vals in sorted(by_id.items())},
        "scored_at": scored_at,
    }
    payload.setdefault("meta", {})
    payload["meta"]["oeq_llm_judge_summary"] = summary
    write_json(response_path, payload)
    return summary


def score_response_file_vision(
    response_path: Path,
    *,
    judge_llm: Any,
    processor: Any,
    sampling_params: Any,
    judge_model_id: str,
    batch_size: int,
    save_every: int,
    max_rows: int,
    resolver: Any,
    annotator: Any,
    sc6_max_frames: int,
    skip_existing_vision: bool,
    question_ids: set[str] | None = None,
) -> dict[str, Any]:
    payload = read_json(response_path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"{response_path} must contain tasks")

    targets = collect_targets(tasks, max_rows, question_ids=question_ids)
    if skip_existing_vision:
        targets = [
            row
            for row in targets
            if not (
                row[1].get("llm_judge_mode") == "vision"
                and isinstance(row[1].get("llm_judge_score"), (int, float))
            )
        ]
    scored_at = datetime.now().isoformat(timespec="seconds")
    by_id: dict[str, list[float]] = defaultdict(list)
    parse_fail = 0
    image_fail = 0
    completed_since_save = 0

    for offset in range(0, len(targets), batch_size):
        batch = targets[offset : offset + batch_size]
        requests: list[dict[str, Any]] = []
        runnable: list[tuple[dict[str, Any], int]] = []

        for _, task in batch:
            cand = strip_thinking(str(task.get("model_response", "")))
            is_sc6 = bool(annotator.is_sc6_task(task))
            try:
                images = resolver.load_images(task)
                if is_sc6:
                    images = subsample_images(images, sc6_max_frames)
                if not images:
                    raise RuntimeError("no images loaded")
                req = build_vision_request(processor, task, cand, images, is_sc6=is_sc6)
                requests.append(req)
                runnable.append((task, len(images)))
            except Exception as exc:
                image_fail += 1
                failed = apply_judge_result(
                    task,
                    judge_model_id=judge_model_id,
                    mode="vision",
                    raw="",
                    score=None,
                    rationale="",
                    scored_at=scored_at,
                    num_images=0,
                    image_error=f"{type(exc).__name__}: {exc}",
                )
                if failed:
                    parse_fail += 1
                completed_since_save += 1

        if requests:
            outputs = judge_llm.generate(requests, sampling_params=sampling_params)
            for (task, n_img), output in zip(runnable, outputs):
                raw = output.outputs[0].text.strip() if output.outputs else ""
                score, rationale = parse_judge_score(raw)
                failed = apply_judge_result(
                    task,
                    judge_model_id=judge_model_id,
                    mode="vision",
                    raw=raw,
                    score=score,
                    rationale=rationale,
                    scored_at=scored_at,
                    num_images=n_img,
                )
                if failed:
                    parse_fail += 1
                elif score is not None:
                    by_id[str(task.get("id", ""))].append(score)
                completed_since_save += 1

        if save_every > 0 and completed_since_save >= save_every:
            write_json(response_path, payload)
            completed_since_save = 0
            print(f"  checkpoint {min(offset + batch_size, len(targets))}/{len(targets)}")

    all_scores = [s for vals in by_id.values() for s in vals]
    summary = {
        "response_path": str(response_path.resolve()),
        "judge_model": judge_model_id,
        "judge_mode": "vision",
        "sc6_max_frames": sc6_max_frames,
        "candidate_count": len(targets),
        "scored_count": len(all_scores),
        "parse_fail_count": parse_fail,
        "image_fail_count": image_fail,
        "overall": bucket(all_scores),
        "per_question_id": {qid: bucket(vals) for qid, vals in sorted(by_id.items())},
        "scored_at": scored_at,
    }
    payload.setdefault("meta", {})
    payload["meta"]["oeq_llm_judge_summary"] = summary
    write_json(response_path, payload)
    return summary


def main() -> None:
    args = parse_args()
    args.hf_home = args.hf_home.expanduser()
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    mode = "text" if args.text_only else "vision"
    print(
        f"[load] judge={args.judge_model} mode={mode} tp={args.tensor_parallel_size} "
        f"max_model_len={args.max_model_len}"
    )
    processor = AutoProcessor.from_pretrained(
        args.judge_model,
        cache_dir=str(args.hf_home / "hub"),
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    judge_llm = LLM(
        model=args.judge_model,
        download_dir=str(args.hf_home / "hub"),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    annotator = None
    resolver = None
    if not args.text_only:
        annotator = load_annotator()
        if args.image_backend == "waymo":
            import sys as _sys

            waymo_dir = Path(__file__).resolve().parent.parent / "full_waymo"
            if str(waymo_dir) not in _sys.path:
                _sys.path.insert(0, str(waymo_dir))
            from run_waymo_miniset_model_benchmark import WaymoVisualResolver

            resolver = WaymoVisualResolver(image_cache_size=64, draw_boxes=True)
        else:
            resolver = annotator.VisualResolver(
                formatted_scenes_dir=args.formatted_scenes_dir,
                nuscenes_root=args.nuscenes_root,
                version=args.version,
                image_cache_size=64,
                render_missing_boxes=False,
            )

    selected = set(args.model)
    question_ids = {str(value) for value in args.only_question_id if str(value).strip()}
    summaries: list[dict[str, Any]] = []
    targets: list[tuple[str, Path]] = []
    if args.response_json:
        for response_json in args.response_json:
            response_path = response_json.expanduser().resolve()
            name = response_path.stem.removesuffix("_responses")
            targets.append((name, response_path))
    else:
        for spec in MODEL_SPECS:
            name = str(spec["name"])
            if selected and name not in selected:
                continue
            targets.append((name, args.output_dir / f"{safe_slug(name)}_responses.json"))

    try:
        for name, path in targets:
            if not path.exists():
                print(f"[skip] missing {path}")
                continue
            print(f"[score] {name} -> {path} ({mode})")
            if args.text_only:
                summary = score_response_file_text(
                    path,
                    judge_llm=judge_llm,
                    sampling_params=sampling,
                    judge_model_id=args.judge_model,
                    batch_size=args.batch_size,
                    save_every=args.save_every,
                    max_rows=args.max_rows,
                    question_ids=question_ids or None,
                )
            else:
                assert resolver is not None and annotator is not None
                summary = score_response_file_vision(
                    path,
                    judge_llm=judge_llm,
                    processor=processor,
                    sampling_params=sampling,
                    judge_model_id=args.judge_model,
                    batch_size=args.batch_size,
                    save_every=args.save_every,
                    max_rows=args.max_rows,
                    resolver=resolver,
                    annotator=annotator,
                    sc6_max_frames=args.sc6_max_frames,
                    skip_existing_vision=args.skip_existing_vision,
                    question_ids=question_ids or None,
                )
            summaries.append({"name": name, **summary})
            print(
                f"  scored={summary['scored_count']} parse_fail={summary['parse_fail_count']} "
                f"mean={summary['overall']['mean']}"
                + (
                    f" image_fail={summary.get('image_fail_count')}"
                    if "image_fail_count" in summary
                    else ""
                )
            )
    finally:
        if resolver is not None:
            resolver.close()

    if not summaries:
        raise SystemExit("No files scored by LLM judge.")

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
            existing_by_name[name]["llm_judge_summary"] = {
                k: item[k]
                for k in (
                    "judge_model",
                    "judge_mode",
                    "sc6_max_frames",
                    "candidate_count",
                    "scored_count",
                    "parse_fail_count",
                    "image_fail_count",
                    "overall",
                    "per_question_id",
                    "scored_at",
                )
                if k in item
            }
        benchmark_summary = build_benchmark_summary(
            input_json=args.input_json,
            output_dir=args.output_dir,
            model_specs=[spec for spec in MODEL_SPECS if spec.get("enabled", True)],
            existing_results_by_name=existing_by_name,
        )
        for result in benchmark_summary.get("results", []):
            name = str(result.get("name", ""))
            prior = existing_by_name.get(name, {})
            if "llm_judge_summary" in prior:
                result["llm_judge_summary"] = prior["llm_judge_summary"]
            if "trajectory_summary" in prior:
                result["trajectory_summary"] = prior["trajectory_summary"]
        write_json(summary_path, benchmark_summary)
        print(f"[summary] refreshed {summary_path}")


if __name__ == "__main__":
    main()
