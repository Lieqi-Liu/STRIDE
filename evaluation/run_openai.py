#!/usr/bin/env python3
"""Minimal OpenAI-compatible runner for STRIDE (text+image).

Requires:
  export OPENAI_API_KEY=...
  export STRIDE_WAYMO_IMAGE_ROOT=...          # for waymo_v1
  export STRIDE_NUSCENES_FORMATTED_SCENES=... # for nuscenes_v6

Example:
  python evaluation/run_openai.py \\
      --split nuscenes_v6 \\
      --model gpt-4.1 \\
      --limit 5 \\
      --output runs/gpt41_responses.json
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stride.aggregate import load_tasks
from stride.io import write_json
from stride.mcq import extract_answer_key
from stride.prompts import build_text_prompt
from stride.visual import load_nuscenes_images, load_waymo_images


def apply_model_result(task: dict, raw_text: str, model: str) -> None:
    task["model_response"] = raw_text
    task["model_response_model"] = model
    if str(task.get("question_format", "")).upper() == "MCQ":
        predicted = extract_answer_key(raw_text, task.get("choices") if isinstance(task.get("choices"), dict) else None)
        gt = str(task.get("ground_truth", "")).strip().upper()[:1]
        choices = task.get("choices") if isinstance(task.get("choices"), dict) else {}
        task["model_predicted_option"] = predicted
        task["model_is_correct"] = bool(predicted) and predicted == gt
        task["model_random_baseline"] = 1.0 / max(len(choices), 1)


def image_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="nuscenes_v6")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("pip install openai") from exc

    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    payload, tasks = load_tasks(args.split)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    backend = "waymo" if str(args.split).startswith("waymo") else "nuscenes"
    out_tasks = []
    for i, task in enumerate(tasks):
        if backend == "waymo":
            images = load_waymo_images(task)
        else:
            images = load_nuscenes_images(task)
        prompt = build_text_prompt(task, num_images=len(images))
        content = [{"type": "input_text", "text": prompt}]
        # Prefer Responses API multimodal shape; fall back to chat.completions style.
        try:
            content_chat = [{"type": "text", "text": prompt}]
            for img in images:
                content_chat.append(
                    {"type": "image_url", "image_url": {"url": image_to_data_url(img)}}
                )
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": content_chat}],
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            raw = f"[ERROR] {type(exc).__name__}: {exc}"

        row = dict(task)
        apply_model_result(row, raw, args.model)
        out_tasks.append(row)
        print(f"[{i+1}/{len(tasks)}] {task.get('id')} -> {raw[:80]!r}")

    write_json(
        args.output,
        {
            "meta": {
                "benchmark": "STRIDE",
                "split": args.split,
                "model": args.model,
                "n_tasks": len(out_tasks),
            },
            "tasks": out_tasks,
        },
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
