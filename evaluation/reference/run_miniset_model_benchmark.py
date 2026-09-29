#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANNOTATOR_PATH = REPO_ROOT / "lieqiliu" / "run_fake_full_vlm_batch.py"

DEFAULT_INPUT_JSON = (
    SCRIPT_DIR / "questions_with_answers_full_nuscenes_miniset_review_trimmed_v6.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "miniset_model_benchmark_outputs"
DEFAULT_FORMATTED_SCENES_DIR = Path("/local1/lieqiliu/nuscenes/fullset/formatted_scenes")
DEFAULT_NUSCENES_ROOT = Path("/local1/lieqiliu/nuscenes/fullset")
DEFAULT_HF_HOME = Path("/local1/lieqiliu/huggingface")


# Edit this list when adding/removing benchmark models.
MODEL_SPECS = [
    {
        "name": "qwen2_5_vl_3b",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "backend": "vllm",
        "enabled": True,
        "batch_size": 8,
        "sc6_max_frames": 8,
        "note": "Qwen2.5-VL-3B backbone used by the local OpenEMMA adaptation.",
    },
    {
        "name": "qwen3_vl_8b",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "backend": "vllm",
        "enabled": True,
    },
    {
        "name": "qwen3_vl_30b_a3b",
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "backend": "vllm",
        "enabled": True,
    },
    {
        "name": "qwen3_vl_32b",
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "backend": "vllm",
        "enabled": True,
    },
    {
        "name": "qwen3_vl_8b_thinking",
        "model_id": "Qwen/Qwen3-VL-8B-Thinking",
        "backend": "vllm",
        "enabled": True,
        # Thinking models emit a long chain-of-thought before the final answer.
        "mcq_max_tokens": 4096,
        "oeq_max_tokens": 4096,
        "batch_size": 8,
        "note": "Qwen3-VL thinking variant; answers extracted after </think>.",
    },
    {
        "name": "qwen3_vl_30b_a3b_thinking",
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "backend": "vllm",
        "enabled": True,
        "mcq_max_tokens": 4096,
        "oeq_max_tokens": 4096,
        "batch_size": 4,
        "note": "Qwen3-VL MoE thinking variant; answers extracted after </think>.",
    },
    {
        "name": "qwen3_6_35b_a3b",
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "backend": "vllm",
        "enabled": True,
        "batch_size": 2,
        "local_files_only": False,
        "enable_thinking": False,
        "sc6_max_frames": 8,
        "note": "Qwen3.6 multimodal MoE; requires qwen36 env / vLLM>=0.19. SC-6 subsampled to 8 frames.",
    },
    {
        "name": "qwen3_6_27b",
        "model_id": "Qwen/Qwen3.6-27B",
        "backend": "vllm",
        "enabled": True,
        "batch_size": 2,
        "local_files_only": False,
        "enable_thinking": False,
        "sc6_max_frames": 8,
        "note": "Qwen3.6 multimodal dense; requires qwen36 env / vLLM>=0.19. SC-6 subsampled to 8 frames.",
    },
    {
        "name": "qwen3_5_27b",
        "model_id": "Qwen/Qwen3.5-27B",
        "backend": "vllm",
        "enabled": False,
        "note": (
            "Disabled: Qwen3.5-27B is text-only (not a VLM) and current vLLM/transformers "
            "may not support the qwen3_5 architecture. Re-enable after upgrading the stack."
        ),
    },
    {
        "name": "llava_1_5_13b",
        "model_id": "llava-hf/llava-1.5-13b-hf",
        "backend": "transformers_llava",
        "enabled": True,
        "batch_size": 2,
        "num_frames": 5,
        "local_files_only": False,
        "note": (
            "Uses all 5 consecutive frames via LLaVA 1.5 multi-image chat "
            "(see https://huggingface.co/llava-hf/llava-1.5-13b-hf). "
            "Downloads from Hugging Face if not cached."
        ),
    },
]


@dataclass
class LoadedModel:
    backend: str
    processor: Any
    llm: Any
    mcq_sampling: Any
    oeq_sampling: Any
    model_id: str


def load_annotator_module() -> Any:
    spec = importlib.util.spec_from_file_location("full_nuscenes_vlm_annotator", ANNOTATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load annotator module from {ANNOTATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANNOTATOR = load_annotator_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple VLMs on the selected full-NuScenes miniset and summarize outcomes."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formatted-scenes-dir", type=Path, default=DEFAULT_FORMATTED_SCENES_DIR)
    parser.add_argument("--nuscenes-root", type=Path, default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable vLLM CUDA graphs (default: true; avoids OOM on A6000 for 30B).",
    )
    parser.add_argument("--mcq-max-tokens", type=int, default=8)
    parser.add_argument("--oeq-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=256)
    parser.add_argument(
        "--only-model",
        action="append",
        default=[],
        help="Run only matching MODEL_SPECS name(s). Can be repeated.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only locally cached HF files by default. Use --no-local-files-only to download.",
    )
    parser.add_argument(
        "--only-question-id",
        action="append",
        default=[],
        help="Run only tasks whose id matches (e.g. SC-6). Can be repeated.",
    )
    parser.add_argument(
        "--only-question-format",
        action="append",
        default=[],
        choices=("MCQ", "OEQ", "mcq", "oeq"),
        help="Run only MCQ or OEQ tasks. Can be repeated.",
    )
    parser.add_argument(
        "--resume-from-output",
        action="store_true",
        help="Merge existing per-model response JSON and only rerun selected question ids.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def is_mcq(row: dict[str, Any]) -> bool:
    return str(row.get("question_format", "")).upper() == "MCQ" and isinstance(row.get("choices"), dict)


def is_oeq(row: dict[str, Any]) -> bool:
    return str(row.get("question_format", "")).upper() == "OEQ"


def is_error_response(row: dict[str, Any]) -> bool:
    return str(row.get("model_response", "")).strip().startswith("[ERROR]")


def row_correctness(row: dict[str, Any]) -> bool | None:
    value = row.get("model_is_correct")
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def clear_prior_outputs(task: dict[str, Any]) -> None:
    ANNOTATOR.clear_model_response(task)
    for key in ("bleurt_model_gt", "bleurt_model", "bleurt_scored_at"):
        task.pop(key, None)


def task_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return ANNOTATOR.task_key(task)


def matches_question_filter(task: dict[str, Any], requested_ids: set[str]) -> bool:
    return not requested_ids or str(task.get("id", "")) in requested_ids


def requested_question_formats(args: argparse.Namespace) -> set[str]:
    return {str(value).upper() for value in args.only_question_format if str(value).strip()}


def matches_format_filter(task: dict[str, Any], requested_formats: set[str]) -> bool:
    if not requested_formats:
        return True
    return str(task.get("question_format", "")).upper() in requested_formats


def prepare_payload(
    input_payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = ANNOTATOR.sanitize_payload(input_payload)
    all_tasks = [dict(task) for task in payload.get("tasks", [])]
    if not isinstance(all_tasks, list):
        raise ValueError("Input JSON must contain a top-level tasks list.")

    requested_ids = {str(value) for value in args.only_question_id if str(value).strip()}
    requested_formats = requested_question_formats(args)
    if args.resume_from_output and output_path is not None and output_path.exists():
        existing = read_json(output_path)
        existing_tasks = existing.get("tasks", [])
        if not isinstance(existing_tasks, list):
            raise ValueError(f"{output_path} must contain a top-level tasks list.")
        existing_by_key = {task_key(task): task for task in existing_tasks}
        existing_with_response = sum(
            1 for task in existing_tasks if str(task.get("model_response", "")).strip()
        )
        merged_tasks: list[dict[str, Any]] = []
        matched_keys = 0
        for task in all_tasks:
            key = task_key(task)
            if key in existing_by_key:
                matched_keys += 1
                merged_tasks.append(dict(existing_by_key[key]))
            else:
                merged_tasks.append(dict(task))
        merged_with_response = sum(
            1 for task in merged_tasks if str(task.get("model_response", "")).strip()
        )
        all_tasks = merged_tasks
        payload["meta"] = dict(existing.get("meta", {}))
        payload["meta"]["resume_merge_matched_keys"] = matched_keys
        payload["meta"]["resume_merge_total_tasks"] = len(all_tasks)
        payload["meta"]["resume_merge_existing_responses"] = existing_with_response
        payload["meta"]["resume_merge_kept_responses"] = merged_with_response
        if existing_with_response > 0 and merged_with_response < int(0.8 * existing_with_response):
            print(
                "WARNING: resume merge kept "
                f"{merged_with_response}/{existing_with_response} prior model responses "
                f"({matched_keys}/{len(all_tasks)} task keys matched). "
                "Input tasks may have changed since the saved responses were created.",
                flush=True,
            )

    runnable_tasks = all_tasks
    if requested_ids:
        runnable_tasks = [task for task in runnable_tasks if matches_question_filter(task, requested_ids)]
    if requested_formats:
        runnable_tasks = [task for task in runnable_tasks if matches_format_filter(task, requested_formats)]

    if args.start_index:
        runnable_tasks = runnable_tasks[args.start_index :]
    if args.max_tasks > 0:
        runnable_tasks = runnable_tasks[: args.max_tasks]

    for task in runnable_tasks:
        clear_prior_outputs(task)

    payload["tasks"] = all_tasks
    payload.setdefault("meta", {})
    if requested_ids:
        payload["meta"]["only_question_id"] = sorted(requested_ids)
    if requested_formats:
        payload["meta"]["only_question_format"] = sorted(requested_formats)
    payload["meta"]["sc6_frame_policy"] = "all_40_scene_grid_frames"
    return payload, runnable_tasks


def model_cache_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def resolve_hf_home(model_id: str, hf_home: Path, local_files_only: bool) -> Path:
    hf_home = hf_home.expanduser()
    if (hf_home / "hub" / model_cache_dir_name(model_id)).exists():
        return hf_home

    default_hf_home = Path.home() / ".cache" / "huggingface"
    if (default_hf_home / "hub" / model_cache_dir_name(model_id)).exists():
        return default_hf_home

    if local_files_only:
        raise FileNotFoundError(
            f"Local cache not found for {model_id}. Checked {hf_home} and {default_hf_home}. "
            "Pass --no-local-files-only to download, or set local_files_only: false on the model spec."
        )
    return hf_home


def apply_hf_home_env(hf_home: Path) -> Path:
    """Point HF/vLLM at the resolved cache root (per-model fallback may differ from --hf-home)."""
    hf_home = hf_home.expanduser().resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    return hf_home


def build_llava_task_prompt(task: dict[str, Any], *, num_frames: int = 5) -> str:
    """Plain-text prompt aligned with the NuScenes benchmark convention."""
    masked_ref = ANNOTATOR.MASKED_OBJECT_REFERENCE
    question = str(task.get("question", "")).replace("<obj>", masked_ref)
    choices = task.get("choices", {})
    question_format = str(task.get("question_format", "MCQ")).upper()

    if question_format == "MCQ" and isinstance(choices, dict) and choices:
        answer_instruction = (
            "Select exactly one option from the provided choices. "
            "Respond with only the option key, for example A."
        )
        choice_text = f"Choices: {json.dumps(choices, ensure_ascii=False)}\n"
    else:
        answer_instruction = "Provide a concise plain-text answer."
        choice_text = ""

    if ANNOTATOR.is_sc6_task(task):
        context_line = (
            f"You are given {num_frames} chronological 360-degree multi-camera frames from one driving scene. "
            "Use the full clip to summarize the overall environment.\n"
        )
        target_line = ""
    else:
        target_line = ""
        if task.get("object_id"):
            target_line = f"Target object: {masked_ref} in the {num_frames}th image.\n"
        context_line = (
            f"You are given {num_frames} consecutive driving frames. "
            "The first 4 images are temporal context. "
            f"The {num_frames}th image is the query frame; when a target object exists, it is highlighted by "
            "a red bounding box.\n"
        )

    return (
        f"{context_line}"
        f"Question ID: {task.get('id', '')}\n"
        f"{target_line}"
        f"Question: {question}\n"
        f"{choice_text}"
        f"{answer_instruction}"
    )


def build_llava_conversation(task: dict[str, Any], images: list[Any]) -> list[dict[str, Any]]:
    """HF chat format: one image content entry per frame, then the text prompt."""
    prompt_text = build_llava_task_prompt(task, num_frames=len(images))
    content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def prepare_llava_inputs(processor: Any, images: list[Any], conversation: list[dict[str, Any]]) -> Any:
    """Build model inputs for multi-image LLaVA (transformers >= 4.35.3)."""
    try:
        return processor.apply_chat_template(
            conversation,
            images=images,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        return processor(images=images, text=prompt, return_tensors="pt")


def model_local_files_only(model_spec: dict[str, Any], args: argparse.Namespace) -> bool:
    if "local_files_only" in model_spec:
        return bool(model_spec["local_files_only"])
    return args.local_files_only


def load_vllm_model(
    model_id: str,
    args: argparse.Namespace,
    *,
    local_files_only: bool,
    model_spec: dict[str, Any] | None = None,
) -> LoadedModel:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model_spec = model_spec or {}
    mcq_max_tokens = int(model_spec.get("mcq_max_tokens", args.mcq_max_tokens))
    oeq_max_tokens = int(model_spec.get("oeq_max_tokens", args.oeq_max_tokens))

    hf_home = apply_hf_home_env(resolve_hf_home(model_id, args.hf_home, local_files_only))
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(hf_home / "hub"),
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    llm = LLM(
        model=model_id,
        download_dir=str(hf_home / "hub"),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    mcq_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=mcq_max_tokens,
        repetition_penalty=1.0,
    )
    oeq_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=oeq_max_tokens,
        repetition_penalty=1.0,
    )
    return LoadedModel(
        backend="vllm",
        processor=processor,
        llm=llm,
        mcq_sampling=mcq_sampling,
        oeq_sampling=oeq_sampling,
        model_id=model_id,
    )


def load_transformers_llava_model(
    model_id: str,
    args: argparse.Namespace,
    *,
    local_files_only: bool,
) -> LoadedModel:
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    hf_home = apply_hf_home_env(resolve_hf_home(model_id, args.hf_home, local_files_only))
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(hf_home / "hub"),
        local_files_only=local_files_only,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=str(hf_home / "hub"),
        local_files_only=local_files_only,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    sampling_defaults = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    mcq_sampling = {"max_new_tokens": args.mcq_max_tokens, **sampling_defaults}
    oeq_sampling = {"max_new_tokens": args.oeq_max_tokens, **sampling_defaults}
    return LoadedModel(
        backend="transformers_llava",
        processor=processor,
        llm=model,
        mcq_sampling=mcq_sampling,
        oeq_sampling=oeq_sampling,
        model_id=model_id,
    )


def load_model(
    model_spec: dict[str, Any],
    args: argparse.Namespace,
    *,
    local_files_only: bool | None = None,
) -> LoadedModel:
    backend = str(model_spec.get("backend", ""))
    model_id = str(model_spec["model_id"])
    use_local_only = args.local_files_only if local_files_only is None else local_files_only
    if backend == "vllm":
        return load_vllm_model(
            model_id,
            args,
            local_files_only=use_local_only,
            model_spec=model_spec,
        )
    if backend == "transformers_llava":
        return load_transformers_llava_model(model_id, args, local_files_only=use_local_only)
    raise ValueError(f"Unsupported backend: {backend}")


def iter_batches(indices: list[int], batch_size: int, desc: str):
    iterator = range(0, len(indices), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(indices) + batch_size - 1) // batch_size, desc=desc)
    for offset in iterator:
        yield indices[offset : offset + batch_size]


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


def run_indices_vllm(
    tasks: list[dict[str, Any]],
    indices: list[int],
    loaded: LoadedModel,
    sampling_params: Any,
    resolver: Any,
    model_id: str,
    output_path: Path,
    payload: dict[str, Any],
    args: argparse.Namespace,
    desc: str,
) -> int:
    completed_since_save = 0
    completed = 0
    enable_thinking = getattr(args, "enable_thinking", None)
    sc6_max_frames = int(getattr(args, "sc6_max_frames", 0) or 0)
    for batch_indices in iter_batches(indices, args.batch_size, desc):
        requests: list[dict[str, Any]] = []
        runnable_indices: list[int] = []
        for idx in batch_indices:
            try:
                images = resolver.load_images(tasks[idx])
                if sc6_max_frames and ANNOTATOR.is_sc6_task(tasks[idx]):
                    images = subsample_images(images, sc6_max_frames)
                prompt = ANNOTATOR.build_prompt(
                    loaded.processor,
                    tasks[idx],
                    num_images=len(images),
                    enable_thinking=enable_thinking,
                )
                requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
                runnable_indices.append(idx)
            except Exception as exc:
                ANNOTATOR.apply_result(tasks[idx], f"[ERROR] {type(exc).__name__}: {exc}", model_id)
                completed += 1
                completed_since_save += 1

        if requests:
            outputs = loaded.llm.generate(requests, sampling_params=sampling_params)
            for idx, output in zip(runnable_indices, outputs):
                text = output.outputs[0].text.strip() if output.outputs else ""
                ANNOTATOR.apply_result(tasks[idx], text, model_id)
                completed += 1
                completed_since_save += 1

        if args.save_every > 0 and completed_since_save >= args.save_every:
            payload["meta"]["partial_summary"] = summarize_model(tasks)
            write_json(output_path, payload)
            completed_since_save = 0
    return completed


def run_indices_transformers_llava(
    tasks: list[dict[str, Any]],
    indices: list[int],
    loaded: LoadedModel,
    sampling_params: dict[str, Any],
    resolver: Any,
    model_id: str,
    output_path: Path,
    payload: dict[str, Any],
    args: argparse.Namespace,
    desc: str,
) -> int:
    import torch

    completed_since_save = 0
    completed = 0
    model = loaded.llm
    processor = loaded.processor

    for batch_indices in iter_batches(indices, args.batch_size, desc):
        for idx in batch_indices:
            try:
                images = resolver.load_images(tasks[idx])
                conversation = build_llava_conversation(tasks[idx], images)
                inputs = prepare_llava_inputs(processor, images, conversation)
                device = next(model.parameters()).device
                if hasattr(inputs, "to"):
                    inputs = inputs.to(device)
                elif isinstance(inputs, dict):
                    inputs = {
                        key: value.to(device) if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    }
                input_ids = inputs["input_ids"]
                generate_kwargs = dict(sampling_params)
                max_new_tokens = int(generate_kwargs.pop("max_new_tokens"))
                temperature = float(generate_kwargs.pop("temperature", 0.0))
                top_p = float(generate_kwargs.pop("top_p", 1.0))
                with torch.inference_mode():
                    if temperature > 0:
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    else:
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                generated = output_ids[0, input_ids.shape[1] :]
                text = processor.decode(generated, skip_special_tokens=True).strip()
                ANNOTATOR.apply_result(tasks[idx], text, model_id)
            except Exception as exc:
                ANNOTATOR.apply_result(tasks[idx], f"[ERROR] {type(exc).__name__}: {exc}", model_id)
            completed += 1
            completed_since_save += 1

        if args.save_every > 0 and completed_since_save >= args.save_every:
            payload["meta"]["partial_summary"] = summarize_model(tasks)
            write_json(output_path, payload)
            completed_since_save = 0
    return completed


def run_indices(
    tasks: list[dict[str, Any]],
    indices: list[int],
    loaded: LoadedModel,
    sampling_params: Any,
    resolver: Any,
    model_id: str,
    output_path: Path,
    payload: dict[str, Any],
    args: argparse.Namespace,
    desc: str,
) -> int:
    if loaded.backend == "transformers_llava":
        return run_indices_transformers_llava(
            tasks=tasks,
            indices=indices,
            loaded=loaded,
            sampling_params=sampling_params,
            resolver=resolver,
            model_id=model_id,
            output_path=output_path,
            payload=payload,
            args=args,
            desc=desc,
        )
    return run_indices_vllm(
        tasks=tasks,
        indices=indices,
        loaded=loaded,
        sampling_params=sampling_params,
        resolver=resolver,
        model_id=model_id,
        output_path=output_path,
        payload=payload,
        args=args,
        desc=desc,
    )


def bucket_bleurt(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def summarize_model(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, Counter[str]] = defaultdict(Counter)
    oeq_bleurt_by_id: dict[str, list[float]] = defaultdict(list)
    mcq_rows = [row for row in tasks if is_mcq(row)]
    oeq_rows = [row for row in tasks if is_oeq(row)]
    scored_mcq = [row for row in mcq_rows if row_correctness(row) is not None]
    correct_mcq = [row for row in scored_mcq if row_correctness(row) is True]
    random_baselines = [
        float(row["model_random_baseline"])
        for row in scored_mcq
        if isinstance(row.get("model_random_baseline"), (int, float))
    ]

    for row in tasks:
        qid = str(row.get("id", ""))
        counter = by_id[qid]
        counter["total"] += 1
        if row.get("model_response"):
            counter["answered"] += 1
        if is_error_response(row):
            counter["errors"] += 1
        correct = row_correctness(row)
        if correct is not None:
            counter["scored"] += 1
            counter["correct"] += int(correct)
        if is_oeq(row):
            bleurt = row.get("bleurt_model_gt")
            if isinstance(bleurt, (int, float)):
                value = float(bleurt)
                oeq_bleurt_by_id[qid].append(value)
                counter["bleurt_scored"] += 1

    oeq_bleurt_values = [
        float(row["bleurt_model_gt"])
        for row in oeq_rows
        if isinstance(row.get("bleurt_model_gt"), (int, float))
    ]
    bleurt_model = next(
        (
            str(row["bleurt_model"])
            for row in oeq_rows
            if isinstance(row.get("bleurt_model"), str) and row.get("bleurt_model")
        ),
        None,
    )

    per_question_id = {}
    for qid, counter in sorted(by_id.items()):
        entry = {
            "total": counter["total"],
            "answered": counter["answered"],
            "error_responses": counter["errors"],
            "scored_accuracy_rows": counter["scored"],
            "correct": counter["correct"],
            "accuracy": counter["correct"] / counter["scored"] if counter["scored"] else None,
        }
        bleurt_bucket = bucket_bleurt(oeq_bleurt_by_id.get(qid, []))
        if bleurt_bucket["count"]:
            entry["bleurt_count"] = bleurt_bucket["count"]
            entry["bleurt_mean"] = bleurt_bucket["mean"]
            entry["bleurt_min"] = bleurt_bucket["min"]
            entry["bleurt_max"] = bleurt_bucket["max"]
        per_question_id[qid] = entry

    mcq_accuracy = len(correct_mcq) / len(scored_mcq) if scored_mcq else None
    random_baseline = sum(random_baselines) / len(random_baselines) if random_baselines else None
    oeq_responses = sum(1 for row in oeq_rows if row.get("model_response") and not is_error_response(row))
    return {
        "total": len(tasks),
        "by_question_format": dict(sorted(Counter(row.get("question_format") for row in tasks).items())),
        "answered": sum(1 for row in tasks if row.get("model_response")),
        "error_responses": sum(1 for row in tasks if is_error_response(row)),
        "mcq": {
            "total": len(mcq_rows),
            "scored": len(scored_mcq),
            "correct": len(correct_mcq),
            "accuracy": mcq_accuracy,
            "random_guess_baseline": random_baseline,
            "accuracy_minus_random_guess": mcq_accuracy - random_baseline
            if mcq_accuracy is not None and random_baseline is not None
            else None,
        },
        "oeq": {
            "total": len(oeq_rows),
            "responses": oeq_responses,
            "bleurt": bucket_bleurt(oeq_bleurt_values),
            "bleurt_model": bleurt_model,
            "by_question_id": {
                qid: bucket_bleurt(values) for qid, values in sorted(oeq_bleurt_by_id.items()) if values
            },
        },
        "per_question_id": per_question_id,
    }


def cleanup_loaded_model(loaded: LoadedModel | None) -> None:
    if loaded is None:
        return
    del loaded
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def merge_runnable_results(
    all_tasks: list[dict[str, Any]],
    runnable_tasks: list[dict[str, Any]],
) -> None:
    runnable_by_key = {task_key(task): task for task in runnable_tasks}
    for index, task in enumerate(all_tasks):
        key = task_key(task)
        if key in runnable_by_key:
            all_tasks[index] = runnable_by_key[key]


def run_model_spec(
    model_spec: dict[str, Any],
    input_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = str(model_spec["name"])
    model_id = str(model_spec["model_id"])
    output_path = args.output_dir / f"{safe_slug(name)}_responses.json"
    started_at = datetime.now().isoformat(timespec="seconds")
    payload, runnable_tasks = prepare_payload(input_payload, args, output_path=output_path)
    tasks = payload["tasks"]
    payload["meta"]["benchmark_model"] = model_spec
    payload["meta"]["benchmark_started_at"] = started_at

    run_args = argparse.Namespace(**vars(args))
    if model_spec.get("batch_size"):
        run_args.batch_size = int(model_spec["batch_size"])
    if "enable_thinking" in model_spec:
        run_args.enable_thinking = bool(model_spec["enable_thinking"])
    if model_spec.get("sc6_max_frames"):
        run_args.sc6_max_frames = int(model_spec["sc6_max_frames"])
    has_sc6 = any(ANNOTATOR.is_sc6_task(task) for task in runnable_tasks)
    if has_sc6 and not getattr(run_args, "sc6_max_frames", 0):
        run_args.max_model_len = max(run_args.max_model_len, ANNOTATOR.SC6_MAX_MODEL_LEN)
        run_args.batch_size = min(run_args.batch_size, 2)

    use_local_only = model_local_files_only(model_spec, run_args)

    loaded: LoadedModel | None = None
    try:
        if not use_local_only:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        loaded = load_model(model_spec, run_args, local_files_only=use_local_only)
    except Exception as exc:
        summary = {
            "name": name,
            "model_id": model_id,
            "backend": model_spec.get("backend"),
            "status": "skipped_load_failed",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "load_error": f"{type(exc).__name__}: {exc}",
            "load_traceback": traceback.format_exc(limit=8),
            "output_json": None,
        }
        return summary
    finally:
        if args.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        resolver = ANNOTATOR.VisualResolver(
            formatted_scenes_dir=args.formatted_scenes_dir,
            nuscenes_root=args.nuscenes_root,
            version=args.version,
            image_cache_size=64,
            render_missing_boxes=True,
        )
        try:
            requested_ids = {str(value) for value in args.only_question_id if str(value).strip()}
            requested_formats = requested_question_formats(run_args)
            runnable_by_key = {task_key(task): task for task in runnable_tasks}
            mcq_indices = [
                idx
                for idx, task in enumerate(runnable_tasks)
                if is_mcq(task)
                and matches_question_filter(task, requested_ids)
                and matches_format_filter(task, requested_formats)
            ]
            oeq_indices = [
                idx
                for idx, task in enumerate(runnable_tasks)
                if is_oeq(task)
                and matches_question_filter(task, requested_ids)
                and matches_format_filter(task, requested_formats)
            ]
            run_indices(
                tasks=runnable_tasks,
                indices=mcq_indices,
                loaded=loaded,
                sampling_params=loaded.mcq_sampling,
                resolver=resolver,
                model_id=model_id,
                output_path=output_path,
                payload=payload,
                args=run_args,
                desc=f"{name} MCQ",
            )
            run_indices(
                tasks=runnable_tasks,
                indices=oeq_indices,
                loaded=loaded,
                sampling_params=loaded.oeq_sampling,
                resolver=resolver,
                model_id=model_id,
                output_path=output_path,
                payload=payload,
                args=run_args,
                desc=f"{name} OEQ",
            )
        finally:
            resolver.close()

        merge_runnable_results(tasks, runnable_tasks)
        model_summary = summarize_model(tasks)
        payload["meta"]["benchmark_finished_at"] = datetime.now().isoformat(timespec="seconds")
        payload["meta"]["benchmark_summary"] = model_summary
        write_json(output_path, payload)
        return {
            "name": name,
            "model_id": model_id,
            "backend": model_spec.get("backend"),
            "status": "completed",
            "started_at": started_at,
            "finished_at": payload["meta"]["benchmark_finished_at"],
            "output_json": str(output_path),
            "summary": model_summary,
        }
    except Exception as exc:
        write_json(output_path, payload)
        return {
            "name": name,
            "model_id": model_id,
            "backend": model_spec.get("backend"),
            "status": "failed_during_inference",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "inference_error": f"{type(exc).__name__}: {exc}",
            "inference_traceback": traceback.format_exc(limit=8),
            "output_json": str(output_path),
            "partial_summary": summarize_model(tasks),
        }
    finally:
        cleanup_loaded_model(loaded)


def selected_model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return enabled specs in reverse MODEL_SPECS order (last listed runs first)."""
    requested = set(args.only_model)
    specs = []
    for spec in reversed(MODEL_SPECS):
        if not spec.get("enabled", True):
            continue
        if requested and spec["name"] not in requested:
            continue
        specs.append(spec)
    return specs


def build_benchmark_summary(
    *,
    input_json: Path,
    output_dir: Path,
    model_specs: list[dict[str, Any]] | None = None,
    last_run_model_specs: list[dict[str, Any]] | None = None,
    existing_results_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    specs = model_specs or [spec for spec in MODEL_SPECS if spec.get("enabled", True)]
    merged_by_name = dict(existing_results_by_name or {})
    for spec in specs:
        name = str(spec["name"])
        output_path = output_dir / f"{safe_slug(name)}_responses.json"
        if not output_path.exists():
            continue
        payload = read_json(output_path)
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        prior = merged_by_name.get(name, {})
        merged_by_name[name] = {
            "name": name,
            "model_id": str(spec.get("model_id", prior.get("model_id", ""))),
            "backend": spec.get("backend", prior.get("backend")),
            "status": prior.get("status", "completed"),
            "started_at": prior.get("started_at") or payload.get("meta", {}).get("benchmark_started_at"),
            "finished_at": prior.get("finished_at") or payload.get("meta", {}).get("benchmark_finished_at"),
            "output_json": str(output_path),
            "summary": summarize_model(tasks),
        }
        if prior.get("load_error"):
            merged_by_name[name]["load_error"] = prior["load_error"]
            merged_by_name[name]["status"] = prior.get("status", "skipped_load_failed")
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "model_specs": specs,
        "last_run_model_specs": last_run_model_specs or [],
        "results": [merged_by_name[name] for name in sorted(merged_by_name)],
    }


def merge_existing_results(
    summary_path: Path,
    new_results: list[dict[str, Any]],
    enabled_model_names: set[str],
) -> list[dict[str, Any]]:
    merged_by_name: dict[str, dict[str, Any]] = {}
    if summary_path.exists():
        try:
            existing = read_json(summary_path)
            for result in existing.get("results", []):
                name = str(result.get("name", ""))
                if name in enabled_model_names:
                    merged_by_name[name] = result
        except Exception as exc:
            print(f"[warn] Could not merge existing benchmark summary: {type(exc).__name__}: {exc}")

    for result in new_results:
        name = str(result.get("name", ""))
        if name in enabled_model_names:
            merged_by_name[name] = result

    return [merged_by_name[name] for name in sorted(merged_by_name)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.hf_home = args.hf_home.expanduser()
    apply_hf_home_env(args.hf_home)
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    input_payload = read_json(args.input_json)
    specs = selected_model_specs(args)
    results = []
    for spec in specs:
        print(f"\n=== Benchmarking {spec['name']} ({spec['model_id']}) ===", flush=True)
        result = run_model_spec(spec, input_payload, args)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"load_traceback", "inference_traceback"}}, indent=2), flush=True)

    summary_path = args.output_dir / "benchmark_summary.json"
    enabled_specs = [spec for spec in MODEL_SPECS if spec.get("enabled", True)]
    enabled_model_names = {str(spec["name"]) for spec in enabled_specs}
    merged_results = merge_existing_results(summary_path, results, enabled_model_names)
    existing_by_name = {str(result.get("name", "")): result for result in merged_results}
    benchmark_summary = build_benchmark_summary(
        input_json=args.input_json,
        output_dir=args.output_dir,
        model_specs=enabled_specs,
        last_run_model_specs=specs,
        existing_results_by_name=existing_by_name,
    )
    write_json(summary_path, benchmark_summary)
    print(f"\nWrote benchmark summary: {summary_path}")


if __name__ == "__main__":
    main()
