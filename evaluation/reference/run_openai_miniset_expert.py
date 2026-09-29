#!/usr/bin/env python3
"""Run OpenAI proprietary VLMs (e.g. gpt-5.5) on the full nuScenes miniset.

Auth (do not commit keys):
  export OPENAI_API_KEY='sk-...'
  # optional lab proxy only:
  # export OPENAI_BASE_URL='https://api.openai.com/v1'

Uses the same VisualResolver grids + MCQ parsing as other miniset VLMs.
"""
from __future__ import annotations

import argparse
import base64
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "miniset_model_benchmark_outputs" / "proprietary_openai_v6"
DEFAULT_FORMATTED = Path("/local1/lieqiliu/nuscenes/fullset/formatted_scenes")
DEFAULT_NUSCENES = Path("/local1/lieqiliu/nuscenes/fullset")
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_KEY_FILE = Path.home() / ".config" / "openai" / "api_key"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--formatted-scenes-dir", type=Path, default=DEFAULT_FORMATTED)
    p.add_argument("--nuscenes-root", type=Path, default=DEFAULT_NUSCENES)
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model id, e.g. gpt-5.5")
    p.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--question-id", action="append", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--save-every", type=int, default=10)
    # GPT-5.x spends completion tokens on hidden reasoning; 64/512 often yields empty text.
    p.add_argument("--max-output-tokens-mcq", type=int, default=4096)
    p.add_argument("--max-output-tokens-oeq", type=int, default=4096)
    p.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Cap SC-6 frames (Qwen miniset uses all 40 scene grids).",
    )
    p.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    p.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="low",
        help="GPT-5.x reasoning.effort via chat.completions reasoning_effort (cuts empty answers).",
    )
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests (rate limit).")
    p.add_argument("--max-retries", type=int, default=6)
    return p.parse_args()


def load_annotator() -> Any:
    spec = importlib.util.spec_from_file_location("full_nuscenes_vlm_annotator", ANNOTATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load annotator from {ANNOTATOR_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_api_key(key_file: Path) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            return key
    raise SystemExit(
        "Missing OpenAI API key. Either:\n"
        "  export OPENAI_API_KEY='sk-...'\n"
        f"  or put the key in {key_file} (chmod 600)\n"
    )


def build_instruction(task: dict[str, Any], num_images: int, annotator: Any) -> str:
    question = str(task.get("question", "")).replace("<obj>", annotator.MASKED_OBJECT_REFERENCE)
    choices = task.get("choices", {})
    question_format = str(task.get("question_format", "MCQ"))

    if question_format == "MCQ" and isinstance(choices, dict) and choices:
        answer_instruction = (
            "Select exactly one option from the provided choices. "
            "Respond with only the option key, for example A."
        )
        choice_text = f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
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


def image_to_data_url(image: Image.Image, detail: str) -> dict[str, Any]:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
    }


def build_messages(frames: list[Image.Image], instruction: str, detail: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [image_to_data_url(im, detail) for im in frames]
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def create_client(api_key: str, base_url: str | None):
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def generate_one(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    max_retries: int,
    reasoning_effort: str | None = "low",
) -> str:
    """Call chat.completions; tolerate GPT-5 param naming differences."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_output_tokens,
            }
            if reasoning_effort:
                create_kwargs["reasoning_effort"] = reasoning_effort
            try:
                resp = client.chat.completions.create(**create_kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                # Older SDKs / models may still want max_tokens.
                if "max_completion_tokens" in msg or "unexpected" in msg:
                    fallback = {k: v for k, v in create_kwargs.items() if k != "max_completion_tokens"}
                    fallback["max_tokens"] = max_output_tokens
                    resp = client.chat.completions.create(**fallback)
                elif reasoning_effort and "reasoning_effort" in msg:
                    # Model/endpoint may not support the knob; retry without it.
                    create_kwargs.pop("reasoning_effort", None)
                    resp = client.chat.completions.create(**create_kwargs)
                else:
                    raise
            choice = resp.choices[0].message
            text = (choice.content or "").strip()
            if not text and getattr(choice, "refusal", None):
                text = f"[REFUSAL] {choice.refusal}"
            if not text:
                # GPT-5.x sometimes burns the whole budget on reasoning and returns "".
                raise RuntimeError(
                    "empty_model_content: completion had no visible text "
                    "(likely max_completion_tokens too low for reasoning)"
                )
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
                )
            )
            if not retryable or attempt + 1 >= max_retries:
                raise
            sleep_s = min(60.0, 2.0 ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(f"generate_one failed: {last_err}")


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key_file)
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
        # Re-run blanks too: gpt-5.5 often returns "" when the reasoning budget is exhausted.
        runnable_idx = [
            i
            for i in runnable_idx
            if (
                not str(tasks[i].get("model_response", "")).strip()
                or str(tasks[i].get("model_response", "")).startswith("[ERROR]")
                or str(tasks[i].get("model_response", "")).startswith("[REFUSAL]")
            )
        ]
    if args.limit and args.limit > 0:
        runnable_idx = runnable_idx[: args.limit]

    print(f"[OpenAI] model={args.model} remaining={len(runnable_idx)}/{len(tasks)}")
    print(f"[OpenAI] reasoning_effort={args.reasoning_effort}")
    print(f"[OpenAI] base_url={args.base_url or 'https://api.openai.com/v1 (default)'}")
    client = create_client(api_key, args.base_url)

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
            "backend": "openai_chat_completions",
            "image_detail": args.image_detail,
            "max_frames": args.max_frames,
            "reasoning_effort": args.reasoning_effort,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": args.base_url or "https://api.openai.com/v1",
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
                messages = build_messages(frames, instruction, args.image_detail)
                max_out = (
                    args.max_output_tokens_mcq
                    if str(task.get("question_format")) == "MCQ"
                    else args.max_output_tokens_oeq
                )
                raw = generate_one(
                    client,
                    model=args.model,
                    messages=messages,
                    max_output_tokens=max_out,
                    max_retries=args.max_retries,
                    reasoning_effort=args.reasoning_effort,
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
    print(f"[OpenAI] wrote {out_path}")
    resolver.close()


if __name__ == "__main__":
    main()
