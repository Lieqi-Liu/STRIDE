#!/usr/bin/env python3
"""Run Dolphins (SaFo-Lab / gray311) on the full nuScenes miniset.

Follows the official inference API from:
  https://github.com/SaFo-Lab/Dolphins/blob/main/inference.py

Prompt format (unchanged from upstream):
  USER: <image> is a driving video. {instruction} GPT:<answer>

Vision input: the same multi-camera grid frames used by other miniset VLMs
(VisualResolver), stacked as a video tensor (F frames) for OpenFlamingo.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from PIL import Image
from tqdm import tqdm

DOLPHINS_ROOT = Path("/data2/lieqi/expert_models/Dolphins")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANNOTATOR_PATH = REPO_ROOT / "lieqiliu" / "run_fake_full_vlm_batch.py"

DEFAULT_INPUT_JSON = SCRIPT_DIR / "questions_with_answers_full_nuscenes_miniset_review_trimmed_v6.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "miniset_model_benchmark_outputs" / "expert_dolphins_v6"
DEFAULT_FORMATTED = Path("/local1/lieqiliu/nuscenes/fullset/formatted_scenes")
DEFAULT_NUSCENES = Path("/local1/lieqiliu/nuscenes/fullset")
DEFAULT_HF_HOME = Path("/data2/lieqi/huggingface")
MODEL_NAME = "dolphins_gray311"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--formatted-scenes-dir", type=Path, default=DEFAULT_FORMATTED)
    p.add_argument("--nuscenes-root", type=Path, default=DEFAULT_NUSCENES)
    p.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    p.add_argument("--dolphins-root", type=Path, default=DOLPHINS_ROOT)
    p.add_argument("--limit", type=int, default=0, help="If >0, only run first N runnable tasks (smoke).")
    p.add_argument("--question-id", action="append", default=None, help="Filter to these IDs (repeatable).")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--max-new-tokens-mcq", type=int, default=64)
    p.add_argument("--max-new-tokens-frq", type=int, default=512)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=16, help="Cap SC-6 frames (official demo uses 16).")
    return p.parse_args()


def load_annotator() -> Any:
    spec = importlib.util.spec_from_file_location("full_nuscenes_vlm_annotator", ANNOTATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load annotator from {ANNOTATOR_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_dolphins(dolphins_root: Path, hf_home: Path):
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home))
    sys.path.insert(0, str(dolphins_root))

    from configs.lora_config import openflamingo_tuning_config
    from huggingface_hub import hf_hub_download
    from mllm.src.factory import create_model_and_transforms

    peft_config = LoraConfig(**openflamingo_tuning_config)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14-336",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=4,
        use_peft=True,
        peft_config=peft_config,
        cache_dir=str(hf_home),
        clip_vision_encoder_cache_dir=str(hf_home),
    )
    checkpoint_path = hf_hub_download("gray311/Dolphins", "checkpoint.pt", cache_dir=str(hf_home))
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.half().cuda().eval()
    return model, image_processor, tokenizer, checkpoint_path


def build_instruction(task: dict[str, Any], num_images: int, annotator: Any) -> str:
    """Same task text as other miniset VLMs, without chat-template wrapping."""
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


def strip_dolphin_answer(decoded: str) -> str:
    """Keep the completion after the official GPT:<answer> marker when present."""
    text = decoded.strip()
    for marker in ("GPT:<answer>", "GPT: <answer>", "<answer>"):
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
    # Drop trailing special tokens if the tokenizer left them in.
    text = re.sub(r"<\|endofchunk\|>|<\|endoftext\|>|<PAD>", "", text).strip()
    return text


@torch.inference_mode()
def generate_one(
    model,
    image_processor,
    tokenizer,
    frames: list[Image.Image],
    instruction: str,
    *,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    vision_x = (
        torch.stack([image_processor(im) for im in frames], dim=0)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1, 1, F, C, H, W) — matches official inference.py
    prompt = [f"USER: <image> is a driving video. {instruction} GPT:<answer>"]
    inputs = tokenizer(prompt, return_tensors="pt")
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": 1,
        "top_k": 0,
        "top_p": 1,
        "no_repeat_ngram_size": 3,
        "length_penalty": 1,
        "do_sample": False,
        "early_stopping": True,
    }
    generated_tokens = model.generate(
        vision_x=vision_x.half().cuda(),
        lang_x=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
        num_beams=num_beams,
        **generation_kwargs,
    )
    if isinstance(generated_tokens, tuple):
        generated_tokens = generated_tokens[0]
    decoded = tokenizer.batch_decode(generated_tokens.cpu())[0]
    return strip_dolphin_answer(decoded)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{MODEL_NAME}_responses.json"
    log_path = args.output_dir / "dolphins_infer.log"

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
            if not str(tasks[i].get("model_response", "")).strip()
            or str(tasks[i].get("model_response", "")).startswith("[ERROR]")
        ]
    if args.limit and args.limit > 0:
        runnable_idx = runnable_idx[: args.limit]

    print(f"[Dolphins] loading model… remaining={len(runnable_idx)}/{len(tasks)}")
    model, image_processor, tokenizer, ckpt = load_dolphins(args.dolphins_root, args.hf_home)
    print(f"[Dolphins] checkpoint={ckpt}")

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
            "expert_model": MODEL_NAME,
            "expert_repo": "https://github.com/SaFo-Lab/Dolphins",
            "expert_checkpoint": "gray311/Dolphins/checkpoint.pt",
            "prompt_template": "USER: <image> is a driving video. {instruction} GPT:<answer>",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "max_frames": args.max_frames,
        }
    )

    done_since_save = 0
    with log_path.open("a") as logf:
        for n, idx in enumerate(tqdm(runnable_idx, desc="Dolphins", unit="task")):
            task = tasks[idx]
            try:
                frames = resolver.load_images(task)
                if len(frames) > args.max_frames:
                    # Uniform subsample to official-ish video length for SC-6.
                    step = len(frames) / float(args.max_frames)
                    frames = [frames[int(i * step)] for i in range(args.max_frames)]
                instruction = build_instruction(task, len(frames), annotator)
                max_new = (
                    args.max_new_tokens_mcq
                    if str(task.get("question_format")) == "MCQ"
                    else args.max_new_tokens_frq
                )
                raw = generate_one(
                    model,
                    image_processor,
                    tokenizer,
                    frames,
                    instruction,
                    max_new_tokens=max_new,
                    num_beams=args.num_beams,
                )
                annotator.apply_result(task, raw, MODEL_NAME)
            except Exception as exc:
                err = f"[ERROR] {type(exc).__name__}: {exc}"
                annotator.apply_result(task, err, MODEL_NAME)
                logf.write(f"task={idx} id={task.get('id')} scene={task.get('scene_id')} {err}\n")
                logf.write(traceback.format_exc() + "\n")
                logf.flush()

            done_since_save += 1
            if done_since_save >= args.save_every or n + 1 == len(runnable_idx):
                with out_path.open("w") as f:
                    json.dump({"meta": payload["meta"], "tasks": tasks}, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                done_since_save = 0

    summary = annotator.output_summary(tasks, MODEL_NAME)
    payload["meta"]["run_summary"] = summary
    with out_path.open("w") as f:
        json.dump({"meta": payload["meta"], "tasks": tasks}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"[Dolphins] wrote {out_path}")
    resolver.close()


if __name__ == "__main__":
    main()
