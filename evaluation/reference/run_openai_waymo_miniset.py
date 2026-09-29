#!/usr/bin/env python3
"""Run OpenAI VLMs on the Waymo miniset using the same API settings as nuScenes v6.

Settings copied from full_nuscenes/run_openai_miniset_expert.py:
  model=gpt-5.5, reasoning_effort=low, image_detail=low,
  max_output_tokens MCQ/FRQ=4096, save_every=10, resume, max_retries=6.
Images come from Waymo CAM_FRONT paths with red-box overlay.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
NUSCENES_DIR = SCRIPT_DIR.parent / "full_nuscenes"
if str(NUSCENES_DIR) not in sys.path:
    sys.path.insert(0, str(NUSCENES_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_openai_miniset_expert as openai_nusc  # noqa: E402
from run_waymo_miniset_model_benchmark import (  # noqa: E402
    WaymoVisualResolver,
    apply_trajectory_prompt,
)

openai_nusc.DEFAULT_INPUT_JSON = SCRIPT_DIR / "questions_with_answers_waymo_miniset_50_per_id_v1.json"
openai_nusc.DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "miniset_model_benchmark_outputs" / "proprietary_openai_v1"


def main() -> None:
    args = openai_nusc.parse_args()
    api_key = openai_nusc.resolve_api_key(args.api_key_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_").replace(" ", "_")
    out_path = args.output_dir / f"{safe_name}_responses.json"
    log_path = args.output_dir / f"{safe_name}_infer.log"

    annotator = openai_nusc.load_annotator()
    with args.input_json.open() as f:
        payload = json.load(f)
    tasks: list[dict[str, Any]] = payload["tasks"]

    if out_path.exists() and args.resume:
        existing = json.load(out_path.open())
        by_key = {
            (t.get("scene_id"), t.get("id"), t.get("group_id"), t.get("object_id"), t.get("question_id")): t
            for t in existing.get("tasks", [])
        }
        for t in tasks:
            key = (t.get("scene_id"), t.get("id"), t.get("group_id"), t.get("object_id"), t.get("question_id"))
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
                or str(tasks[i].get("model_response", "")).startswith("[REFUSAL]")
            )
        ]
    if args.limit and args.limit > 0:
        runnable_idx = runnable_idx[: args.limit]

    print(f"[OpenAI] split=waymo_miniset_50_per_id_v1")
    print(f"[OpenAI] model={args.model} remaining={len(runnable_idx)}/{len(tasks)}")
    print(f"[OpenAI] reasoning_effort={args.reasoning_effort} image_detail={args.image_detail}")
    print(f"[OpenAI] max_tokens mcq={args.max_output_tokens_mcq} frq={args.max_output_tokens_frq}")
    print(f"[OpenAI] base_url={args.base_url or 'https://api.openai.com/v1 (default)'}")
    client = openai_nusc.create_client(api_key, args.base_url)
    resolver = WaymoVisualResolver(image_cache_size=32, draw_boxes=True)

    payload.setdefault("meta", {})
    payload["meta"].update(
        {
            "dataset_split": "waymo_miniset_50_per_id_v1",
            "expert_model": args.model,
            "backend": "openai_chat_completions",
            "image_detail": args.image_detail,
            "max_frames": args.max_frames,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens_mcq": args.max_output_tokens_mcq,
            "max_output_tokens_frq": args.max_output_tokens_frq,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": args.base_url or "https://api.openai.com/v1",
            "matched_nuscenes_openai_settings": True,
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
                instruction = apply_trajectory_prompt(
                    task, openai_nusc.build_instruction(task, len(frames), annotator)
                )
                messages = openai_nusc.build_messages(frames, instruction, args.image_detail)
                max_out = (
                    args.max_output_tokens_mcq
                    if str(task.get("question_format")) == "MCQ"
                    else args.max_output_tokens_frq
                )
                raw = openai_nusc.generate_one(
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
