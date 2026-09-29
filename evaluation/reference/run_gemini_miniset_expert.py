#!/usr/bin/env python3
"""Run Google Gemini VLMs on the full nuScenes miniset via Vertex AI (ADC).

Auth:
  gcloud auth application-default login
  # optional:
  # export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
  # export GOOGLE_CLOUD_QUOTA_PROJECT=your-project

Gemini 3.x models require location=global on this project.
Same VisualResolver grids + MCQ parsing as OpenAI / Qwen runners.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANNOTATOR_PATH = REPO_ROOT / "lieqiliu" / "run_fake_full_vlm_batch.py"

DEFAULT_INPUT_JSON = SCRIPT_DIR / "questions_with_answers_full_nuscenes_miniset_review_trimmed_v6.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "miniset_model_benchmark_outputs" / "proprietary_gemini_v6"
DEFAULT_FORMATTED = Path("/local1/lieqiliu/nuscenes/fullset/formatted_scenes")
DEFAULT_NUSCENES = Path("/local1/lieqiliu/nuscenes/fullset")
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "brcl-cstm-r-seas-cs-alpha")
DEFAULT_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--formatted-scenes-dir", type=Path, default=DEFAULT_FORMATTED)
    p.add_argument("--nuscenes-root", type=Path, default=DEFAULT_NUSCENES)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--location", default=DEFAULT_LOCATION, help="Use global for Gemini 3.x")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--question-id", action="append", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--max-output-tokens-mcq", type=int, default=4096)
    p.add_argument("--max-output-tokens-frq", type=int, default=4096)
    p.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Cap SC-6 frames (same default as GPT / Qwen miniset).",
    )
    p.add_argument(
        "--max-image-side",
        type=int,
        default=768,
        help="Resize long image side (approx OpenAI image_detail=low token budget).",
    )
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument(
        "--thinking-level",
        default="low",
        choices=("minimal", "low", "medium", "high"),
        help="Gemini 3 thinking level when supported (maps to GPT reasoning_effort=low).",
    )
    return p.parse_args()


def load_annotator() -> Any:
    spec = importlib.util.spec_from_file_location("full_nuscenes_vlm_annotator", ANNOTATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load annotator from {ANNOTATOR_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_instruction(task: dict[str, Any], num_images: int, annotator: Any) -> str:
    question = str(task.get("question", "")).replace("<obj>", annotator.MASKED_OBJECT_REFERENCE)
    choices = task.get("choices", {})
    question_format = str(task.get("question_format", "MCQ"))

    qid = str(task.get("id", ""))
    if question_format == "MCQ" and isinstance(choices, dict) and choices:
        answer_instruction = (
            "Select exactly one option from the provided choices. "
            "Respond with only the option key, for example A."
        )
        choice_text = f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
    elif qid in {"TRJ-5", "TRJ-6"}:
        answer_instruction = (
            "Return ONLY a JSON array of [x, y] waypoints (meters), e.g. [[1.0, 0.0], [2.0, 0.1]]. "
            "No markdown fences, no extra text."
        )
        choice_text = ""
    else:
        answer_instruction = "Provide a concise plain-text answer."
        choice_text = ""

    if annotator.is_sc6_task(task):
        context_line = (
            f"You are given {num_images} chronological 360-degree multi-camera frames from one driving scene. "
            "Use the full clip to summarize the overall environment.\n"
        )
        target_line = ""
    else:
        context_line = (
            f"You are given {num_images} consecutive driving frames. The first 4 images are temporal context. "
            f"The {num_images}th image is the query frame; when a target object exists, it is highlighted by "
            "a red bounding box.\n"
        )
        target_line = ""
        if task.get("object_id"):
            target_line = (
                f"Target object: {annotator.MASKED_OBJECT_REFERENCE} in the {num_images}th image.\n"
            )

    return (
        f"{context_line}"
        f"Question ID: {task.get('id', '')}\n"
        f"{target_line}"
        f"Question: {question}\n"
        f"{choice_text}"
        f"{answer_instruction}"
    )


def image_to_jpeg_bytes(image: Image.Image, *, max_side: int, quality: int) -> bytes:
    im = image.convert("RGB")
    w, h = im.size
    long_side = max(w, h)
    if max_side > 0 and long_side > max_side:
        scale = max_side / float(long_side)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build_contents(
    frames: list[Image.Image],
    instruction: str,
    *,
    max_side: int,
    quality: int,
) -> list[Any]:
    from google.genai import types

    parts: list[Any] = []
    for im in frames:
        parts.append(
            types.Part.from_bytes(
                data=image_to_jpeg_bytes(im, max_side=max_side, quality=quality),
                mime_type="image/jpeg",
            )
        )
    parts.append(instruction)
    return parts


def create_client(project: str, location: str):
    from google import genai

    return genai.Client(vertexai=True, project=project, location=location)


def generate_one(
    client: Any,
    *,
    model: str,
    contents: list[Any],
    max_output_tokens: int,
    max_retries: int,
    thinking_level: str | None,
) -> str:
    from google.genai import types

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            config_kwargs: dict[str, Any] = {"max_output_tokens": max_output_tokens}
            # Gemini 3 supports thinking_config; older models may reject it.
            if thinking_level:
                try:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                except Exception:
                    # Older SDK enum names: ThinkingLevel
                    pass
            config = types.GenerateContentConfig(**config_kwargs)
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if thinking_level and ("thinking" in msg or "unknown" in msg or "invalid" in msg):
                    config = types.GenerateContentConfig(max_output_tokens=max_output_tokens)
                    resp = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                else:
                    raise
            text = (getattr(resp, "text", None) or "").strip()
            if not text:
                # Fallback: stitch candidate parts
                chunks: list[str] = []
                for cand in getattr(resp, "candidates", None) or []:
                    content = getattr(cand, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        t = getattr(part, "text", None)
                        if t:
                            chunks.append(t)
                text = "\n".join(chunks).strip()
            if not text:
                raise RuntimeError("empty_model_content: Gemini returned no visible text")
            return text
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            retryable = any(
                s in msg
                for s in (
                    "rate limit",
                    "429",
                    "timeout",
                    "503",
                    "502",
                    "overloaded",
                    "temporarily",
                    "empty_model_content",
                    "resource exhausted",
                    "quota",
                )
            )
            if not retryable or attempt + 1 >= max_retries:
                raise
            time.sleep(min(60.0, 2.0 ** attempt))
    raise RuntimeError(f"generate_one failed: {last_err}")


def main() -> None:
    args = parse_args()
    if args.project:
        os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", args.project)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_").replace(" ", "_")
    out_path = args.output_dir / f"{safe_name}_responses.json"
    log_path = args.output_dir / f"{safe_name}_infer.log"

    annotator = load_annotator()
    with args.input_json.open() as f:
        payload = json.load(f)
    tasks: list[dict[str, Any]] = payload["tasks"]

    if out_path.exists() and args.resume:
        existing = json.load(out_path.open())
        by_key = {
            (t.get("scene_id"), t.get("id"), t.get("group_id"), t.get("object_id")): t
            for t in existing.get("tasks", [])
        }
        for t in tasks:
            key = (t.get("scene_id"), t.get("id"), t.get("group_id"), t.get("object_id"))
            prev = by_key.get(key)
            if prev and str(prev.get("model_response", "")).strip():
                for k, v in prev.items():
                    if k.startswith("model_") or k in ("predicted_option", "is_correct", "random_baseline"):
                        t[k] = v

    if args.question_id:
        wanted = set(args.question_id)
        runnable_idx = [i for i, t in enumerate(tasks) if t.get("id") in wanted]
    else:
        runnable_idx = list(range(len(tasks)))

    if args.resume:
        runnable_idx = [
            i
            for i in runnable_idx
            if (
                not str(tasks[i].get("model_response", "")).strip()
                or str(tasks[i].get("model_response", "")).startswith("[ERROR]")
            )
        ]
    if args.limit and args.limit > 0:
        runnable_idx = runnable_idx[: args.limit]

    print(
        f"[Gemini] model={args.model} project={args.project} location={args.location} "
        f"remaining={len(runnable_idx)}/{len(tasks)}"
    )
    print(
        f"[Gemini] max_frames={args.max_frames} max_image_side={args.max_image_side} "
        f"thinking_level={args.thinking_level}"
    )
    client = create_client(args.project, args.location)

    resolver = annotator.VisualResolver(
        formatted_scenes_dir=args.formatted_scenes_dir,
        nuscenes_root=args.nuscenes_root,
        version="v1.0-trainval",
        image_cache_size=32,
        render_missing_boxes=True,
    )

    payload.setdefault("meta", {})
    payload["meta"].update(
        {
            "expert_model": args.model,
            "backend": "vertex_gemini",
            "project": args.project,
            "location": args.location,
            "max_frames": args.max_frames,
            "max_image_side": args.max_image_side,
            "jpeg_quality": args.jpeg_quality,
            "thinking_level": args.thinking_level,
            "image_detail_equiv": "low",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_json": str(args.input_json),
        }
    )

    done_since_save = 0
    with log_path.open("a") as logf:
        for n, idx in enumerate(tqdm(runnable_idx, desc=args.model, unit="task")):
            task = tasks[idx]
            try:
                frames = resolver.load_images(task)
                if len(frames) > args.max_frames:
                    step = len(frames) / float(args.max_frames)
                    frames = [frames[int(i * step)] for i in range(args.max_frames)]
                instruction = build_instruction(task, len(frames), annotator)
                contents = build_contents(
                    frames,
                    instruction,
                    max_side=args.max_image_side,
                    quality=args.jpeg_quality,
                )
                max_out = (
                    args.max_output_tokens_mcq
                    if str(task.get("question_format")) == "MCQ"
                    else args.max_output_tokens_frq
                )
                raw = generate_one(
                    client,
                    model=args.model,
                    contents=contents,
                    max_output_tokens=max_out,
                    max_retries=args.max_retries,
                    thinking_level=args.thinking_level,
                )
                annotator.apply_result(task, raw, args.model)
            except Exception as exc:
                err = f"[ERROR] {type(exc).__name__}: {exc}"
                annotator.apply_result(task, err, args.model)
                logf.write(f"task={idx} id={task.get('id')} scene={task.get('scene_id')} {err}\n")
                logf.write(traceback.format_exc() + "\n")
                logf.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)

            done_since_save += 1
            if done_since_save >= args.save_every or n + 1 == len(runnable_idx):
                with out_path.open("w") as f:
                    json.dump({"meta": payload["meta"], "tasks": tasks}, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                done_since_save = 0

    summary = annotator.output_summary(tasks, args.model)
    payload["meta"]["run_summary"] = summary
    with out_path.open("w") as f:
        json.dump({"meta": payload["meta"], "tasks": tasks}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"[Gemini] wrote {out_path}")
    resolver.close()


if __name__ == "__main__":
    main()
