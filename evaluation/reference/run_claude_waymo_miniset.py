#!/usr/bin/env python3
"""Run Anthropic Claude on the Waymo miniset using the same Vertex settings as nuScenes.

Settings copied from full_nuscenes/run_claude_miniset_expert.py:
  model=claude-sonnet-5, location=global, effort=medium,
  max_output_tokens MCQ/OEQ=4096, max_image_side=768, jpeg_quality=85,
  save_every=5 (via shell), resume, max_retries=6.
Images come from Waymo CAM_FRONT paths with red-box overlay.
"""
from __future__ import annotations

import json
import os
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

import run_claude_miniset_expert as claude_nusc  # noqa: E402
from run_waymo_miniset_model_benchmark import (  # noqa: E402
    WaymoVisualResolver,
    apply_trajectory_prompt,
)


def main() -> None:
    args = claude_nusc.parse_args()
    if args.project:
        os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", args.project)
        os.environ.setdefault("ANTHROPIC_VERTEX_PROJECT_ID", args.project)
    if args.location:
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", args.location)
        os.environ.setdefault("CLOUD_ML_REGION", args.location)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_").replace(" ", "_").replace("@", "_")
    out_path = args.output_dir / f"{safe_name}_responses.json"
    log_path = args.output_dir / f"{safe_name}_infer.log"

    annotator = claude_nusc.load_annotator()
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
            )
        ]
    if args.limit and args.limit > 0:
        runnable_idx = runnable_idx[: args.limit]

    print(f"[Claude] split=waymo_miniset")
    print(
        f"[Claude] model={args.model} project={args.project} location={args.location} "
        f"remaining={len(runnable_idx)}/{len(tasks)}"
    )
    print(
        f"[Claude] max_frames={args.max_frames} max_image_side={args.max_image_side} "
        f"effort={args.effort}"
    )
    client = claude_nusc.create_client(args.project, args.location)
    resolver = WaymoVisualResolver(image_cache_size=32, draw_boxes=True)

    payload.setdefault("meta", {})
    payload["meta"].update(
        {
            "dataset_split": "waymo_miniset",
            "expert_model": args.model,
            "backend": "vertex_claude",
            "project": args.project,
            "location": args.location,
            "max_frames": args.max_frames,
            "max_image_side": args.max_image_side,
            "jpeg_quality": args.jpeg_quality,
            "effort": args.effort,
            "image_detail_equiv": "low",
            "max_output_tokens_mcq": args.max_output_tokens_mcq,
            "max_output_tokens_oeq": args.max_output_tokens_oeq,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_json": str(args.input_json),
            "matched_nuscenes_claude_settings": True,
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
                    task, claude_nusc.build_instruction(task, len(frames), annotator)
                )
                messages = claude_nusc.build_messages(
                    frames,
                    instruction,
                    max_side=args.max_image_side,
                    quality=args.jpeg_quality,
                )
                max_out = (
                    args.max_output_tokens_mcq
                    if str(task.get("question_format")) == "MCQ"
                    else args.max_output_tokens_oeq
                )
                raw = claude_nusc.generate_one(
                    client,
                    model=args.model,
                    messages=messages,
                    max_output_tokens=max_out,
                    max_retries=args.max_retries,
                    effort=args.effort,
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
    print(f"[Claude] wrote {out_path}")
    resolver.close()


if __name__ == "__main__":
    main()
