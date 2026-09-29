#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

FULL_NUSCENES_DIR = Path(__file__).resolve().parent.parent / "full_nuscenes"
if str(FULL_NUSCENES_DIR) not in sys.path:
    sys.path.insert(0, str(FULL_NUSCENES_DIR))

from nuscenes_scene_ids import (  # noqa: E402
    load_formatted_scene_lookup,
    resolve_scene_dir as resolve_formatted_scene_dir,
)

from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional; fall back to simple prints.
    tqdm = None

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_HF_HOME = Path("/local1/lieqiliu/huggingface")
MASKED_OBJECT_REFERENCE = "the object in the red bounding box"
MAX_IMAGE_PIXELS = 640 * 640
SC6_MAX_IMAGE_PIXELS = 360 * 360
SC6_QUESTION_ID = "SC-6"
SC6_FRAME_COUNT = 40
MAX_MODEL_LEN = 8192
SC6_MAX_MODEL_LEN = 32768


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Batch-annotate fake-full driving QA rows with VLM model responses. "
            "The prompt masks object_reference as the red-box object to avoid distance leakage."
        )
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=script_dir / "questions_with_answers_fake_full_combined.json",
        help="Input combined questions JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "questions_with_answers_fake_full_combined_qwen3vl8b.json",
        help="Output JSON with model responses.",
    )
    parser.add_argument(
        "--formatted-scenes-dir",
        type=Path,
        default=Path("/home/lieqiliu/AutoDriving/nuscenes_mini/formatted_scenes"),
        help="formatted_scenes directory containing 5-frame grids and group metadata.",
    )
    parser.add_argument(
        "--nuscenes-root",
        type=Path,
        default=Path("/home/lieqiliu/AutoDriving/nuscenes_mini"),
        help="nuScenes root used for rendering missing object-specific red boxes.",
    )
    parser.add_argument("--version", type=str, default="v1.0-mini", help="nuScenes version.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="VLM model name/path.")
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"Hugging Face cache root (default: {DEFAULT_HF_HOME}).",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only locally cached Hugging Face files; disable to allow downloads.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts per vLLM batch.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=MAX_MODEL_LEN,
        help=f"vLLM max model length (default: {MAX_MODEL_LEN}). Lower this if startup OOMs.",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="If >0, run only first N rows.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based task index to start from before applying --max-tasks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --output and skip rows that already have this model response.",
    )
    parser.add_argument(
        "--rerun-errors",
        action="store_true",
        help="When resuming, rerun rows whose existing model_response starts with [ERROR].",
    )
    parser.add_argument(
        "--keep-existing-responses",
        action="store_true",
        help="Do not overwrite existing non-empty model_response values in the input/output JSON.",
    )
    parser.add_argument(
        "--sanitize-only",
        action="store_true",
        help="Only write a sanitized JSON copy; do not load the model.",
    )
    parser.add_argument("--mcq-max-tokens", type=int, default=8, help="Max output tokens for MCQ.")
    parser.add_argument("--oeq-max-tokens", type=int, default=256, help="Max output tokens for OEQ.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=64,
        help="Number of visual contexts to keep in memory.",
    )
    parser.add_argument(
        "--render-missing-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render per-object red-box images when no matching cached render exists.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=128,
        help="Checkpoint output after this many completed rows.",
    )
    parser.add_argument(
        "--only-question-id",
        action="append",
        default=[],
        help="Run only tasks whose id matches (e.g. SC-6). Can be repeated.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_cache_dir_name(model: str) -> str:
    return f"models--{model.replace('/', '--')}"


def resolve_hf_home(model: str, requested_hf_home: Path, local_files_only: bool) -> Path:
    """Prefer requested cache, but use the default HF cache if that is where the model exists."""
    requested_hf_home = requested_hf_home.expanduser()
    requested_model_dir = requested_hf_home / "hub" / model_cache_dir_name(model)
    if requested_model_dir.exists():
        return requested_hf_home

    default_hf_home = Path.home() / ".cache" / "huggingface"
    default_model_dir = default_hf_home / "hub" / model_cache_dir_name(model)
    if default_model_dir.exists():
        print(
            f"[cache] {model} not found under {requested_hf_home}; "
            f"using cached model under {default_hf_home}."
        )
        return default_hf_home

    if local_files_only:
        raise FileNotFoundError(
            f"Local model cache not found for {model}. Checked:\n"
            f"  - {requested_model_dir}\n"
            f"  - {default_model_dir}\n"
            "Disable --local-files-only if you want to download it."
        )
    return requested_hf_home


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def is_sc6_task(task: dict[str, Any]) -> bool:
    return str(task.get("id", "")) == SC6_QUESTION_ID


def all_scene_grid_frames(
    scene_id: Any,
    formatted_scenes_dir: Path,
    *,
    scene_lookup: dict[str, str] | None = None,
) -> list[Path]:
    scene_dir = resolve_formatted_scene_dir(scene_id, formatted_scenes_dir, lookup=scene_lookup)
    if scene_dir is None:
        raise FileNotFoundError(f"Missing formatted scene directory for {scene_id}")
    grids = sorted(scene_dir.glob("*_grid.jpg"))
    if not grids:
        raise FileNotFoundError(f"No *_grid.jpg frames found in {scene_dir}")
    return grids[:SC6_FRAME_COUNT]


def resize_for_vlm(image: Image.Image, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / float(width * height)) ** 0.5
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def sanitize_task(task: dict[str, Any]) -> dict[str, Any]:
    task = dict(task)
    original = task.get("object_reference")
    if original and original != MASKED_OBJECT_REFERENCE:
        task.setdefault("object_reference_original", original)
        task["object_reference"] = MASKED_OBJECT_REFERENCE
    return task


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["tasks"] = [sanitize_task(task) for task in payload.get("tasks", [])]
    output.setdefault("meta", {})
    output["meta"]["object_reference_mask"] = {
        "value": MASKED_OBJECT_REFERENCE,
        "reason": "Avoid leaking distance/relative-position text to the VLM prompt.",
        "original_field": "object_reference_original",
    }
    return output


def task_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task.get("id"),
        task.get("scene_id"),
        task.get("group_id"),
        task.get("object_id"),
        task.get("question"),
        task.get("source_dataset"),
    )


def merge_resume_payload(base: dict[str, Any], resume_payload: dict[str, Any]) -> dict[str, Any]:
    resume_by_key = {task_key(task): task for task in resume_payload.get("tasks", [])}
    merged = dict(base)
    merged_tasks: list[dict[str, Any]] = []
    for task in base.get("tasks", []):
        old = resume_by_key.get(task_key(task))
        merged_tasks.append(dict(old) if old else task)
    merged["tasks"] = merged_tasks
    return merged


def base_group_id(group_id: str | None) -> str | None:
    if not group_id:
        return None
    match = re.match(r"^(group_\d{3})", group_id)
    return match.group(1) if match else group_id


def resolve_group_file(
    task: dict[str, Any],
    formatted_scenes_dir: Path,
    *,
    scene_lookup: dict[str, str] | None = None,
) -> Path | None:
    source_group_file = task.get("source_group_file")
    if source_group_file:
        direct = formatted_scenes_dir / source_group_file
        if direct.exists():
            return direct
        scene_part = Path(source_group_file).parent
        base_group = base_group_id(task.get("group_id"))
        if base_group:
            fallback = formatted_scenes_dir / scene_part / f"{base_group}_vehicle_annotations.json"
            if fallback.exists():
                return fallback

    scene_id = task.get("scene_id")
    base_group = base_group_id(task.get("group_id"))
    if scene_id:
        scene_dir = resolve_formatted_scene_dir(scene_id, formatted_scenes_dir, lookup=scene_lookup)
        if scene_dir is not None:
            if base_group:
                fallback = scene_dir / f"{base_group}_vehicle_annotations.json"
                if fallback.exists():
                    return fallback
            else:
                scene_level_candidates = sorted(scene_dir.glob("group_*_vehicle_annotations.json"))
                if scene_level_candidates:
                    return scene_level_candidates[0]
    return None


def keep_right_panel(image_path: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
        image.crop((width // 2, 0, width, height)).save(image_path, format="JPEG", quality=95)


class VisualResolver:
    def __init__(
        self,
        formatted_scenes_dir: Path,
        nuscenes_root: Path,
        version: str,
        image_cache_size: int,
        render_missing_boxes: bool,
    ) -> None:
        self.formatted_scenes_dir = formatted_scenes_dir
        self.nuscenes_root = nuscenes_root
        self.version = version
        self.image_cache_size = image_cache_size
        self.render_missing_boxes = render_missing_boxes
        self.scene_lookup = load_formatted_scene_lookup(
            formatted_scenes_dir,
            nuscenes_root=nuscenes_root,
            version=version,
        )
        self.group_cache: dict[Path, dict[str, Any]] = {}
        self.image_cache: OrderedDict[tuple[Any, ...], list[Image.Image]] = OrderedDict()
        self._nusc: Any = None
        self._plt: Any = None
        self.warnings: Counter[str] = Counter()

    def close(self) -> None:
        for images in self.image_cache.values():
            for image in images:
                image.close()
        self.image_cache.clear()

    def group_payload(self, group_file: Path) -> dict[str, Any]:
        if group_file not in self.group_cache:
            self.group_cache[group_file] = read_json(group_file)
        return self.group_cache[group_file]

    def get_nusc(self) -> Any:
        if self._nusc is None:
            from nuscenes.nuscenes import NuScenes
            import matplotlib.pyplot as plt

            self._plt = plt
            self._nusc = NuScenes(version=self.version, dataroot=str(self.nuscenes_root), verbose=False)
        return self._nusc

    def render_object(self, scene_dir: Path, base_group: str, object_id: str) -> Path | None:
        if not self.render_missing_boxes:
            return None
        render_dir = scene_dir / "_object_renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_path = render_dir / f"{base_group}_{object_id}_red_bbox.jpg"
        if render_path.exists():
            return render_path
        try:
            self.get_nusc().render_annotation(object_id, out_path=str(render_path))
            if self._plt is not None:
                self._plt.close("all")
            keep_right_panel(render_path)
            return render_path
        except Exception as exc:  # Rendering should not stop the whole evaluation.
            self.warnings[f"render_failed:{type(exc).__name__}"] += 1
            return None

    def resolve_image_paths(self, task: dict[str, Any]) -> list[Path]:
        if is_sc6_task(task):
            return all_scene_grid_frames(
                task.get("scene_id"),
                self.formatted_scenes_dir,
                scene_lookup=self.scene_lookup,
            )

        group_file = resolve_group_file(
            task,
            self.formatted_scenes_dir,
            scene_lookup=self.scene_lookup,
        )
        if group_file is None:
            raise FileNotFoundError(
                f"Cannot resolve group metadata for {task.get('scene_id')}:{task.get('group_id')}"
            )
        payload = self.group_payload(group_file)
        scene_id = payload["scene_id"]
        scene_dir = self.formatted_scenes_dir / scene_id
        frame_indices = payload.get("frame_indices_1based", [])
        if len(frame_indices) != 5:
            raise ValueError(f"Expected 5 frame indices in {group_file}, got {len(frame_indices)}")

        image_paths: list[Path] = []
        for idx in frame_indices:
            matches = sorted(scene_dir.glob(f"{idx:03d}_*_grid.jpg"))
            if not matches:
                raise FileNotFoundError(f"Missing frame image for index {idx} in {scene_dir}")
            image_paths.append(matches[0])

        object_id = task.get("object_id")
        base_group = payload.get("group_id") or base_group_id(task.get("group_id"))
        if object_id and base_group:
            selected_token = payload.get("selected_vehicle_annotation_token")
            selected_path = scene_dir / f"{base_group}_selected_vehicle_render.jpg"
            if object_id == selected_token and selected_path.exists():
                image_paths[4] = selected_path
            else:
                rendered = self.render_object(scene_dir, base_group, str(object_id))
                if rendered is not None and rendered.exists():
                    image_paths[4] = rendered
                elif selected_path.exists():
                    self.warnings["fallback_selected_group_render"] += 1
                    image_paths[4] = selected_path
        return image_paths

    def load_images(self, task: dict[str, Any]) -> list[Image.Image]:
        if is_sc6_task(task):
            visual_key = ("sc6", task.get("scene_id"))
            max_pixels = SC6_MAX_IMAGE_PIXELS
        else:
            visual_key = (
                task.get("scene_id"),
                base_group_id(task.get("group_id")),
                task.get("object_id"),
            )
            max_pixels = MAX_IMAGE_PIXELS
        cached = self.image_cache.get(visual_key)
        if cached is not None:
            self.image_cache.move_to_end(visual_key)
            return cached

        images: list[Image.Image] = []
        for path in self.resolve_image_paths(task):
            image = Image.open(path).convert("RGB")
            images.append(resize_for_vlm(image, max_pixels=max_pixels))

        self.image_cache[visual_key] = images
        self.image_cache.move_to_end(visual_key)
        while len(self.image_cache) > self.image_cache_size:
            _, old_images = self.image_cache.popitem(last=False)
            for image in old_images:
                image.close()
        return images


def build_prompt(
    processor: AutoProcessor,
    task: dict[str, Any],
    *,
    num_images: int | None = None,
    enable_thinking: bool | None = None,
) -> str:
    if num_images is None:
        num_images = SC6_FRAME_COUNT if is_sc6_task(task) else 5
    image_content = [{"type": "image"} for _ in range(num_images)]
    question = task.get("question", "").replace("<obj>", MASKED_OBJECT_REFERENCE)
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

    if is_sc6_task(task):
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
            target_line = f"Target object: {MASKED_OBJECT_REFERENCE} in the {num_images}th image.\n"

    prompt_text = (
        f"{context_line}"
        f"Question ID: {task.get('id', '')}\n"
        f"{target_line}"
        f"Question: {question}\n"
        f"{choice_text}"
        f"{answer_instruction}"
    )
    messages = [{"role": "user", "content": image_content + [{"type": "text", "text": prompt_text}]}]
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = bool(enable_thinking)
    try:
        return processor.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        if enable_thinking is not None:
            template_kwargs["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        return processor.apply_chat_template(messages, **template_kwargs)


def extract_answer_key(raw_text: str, choices: dict[str, Any] | None = None) -> str | None:
    text = raw_text.strip()
    if not text:
        return None

    # Qwen Thinking / R1-style models emit reasoning then a final answer after </think>.
    think_end = "</" + "think>"
    if think_end in text:
        text = text.rsplit(think_end, 1)[-1].strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            answer = payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip().upper()[:1]
    except Exception:
        pass

    normalized = text.strip().upper()
    if len(normalized) == 1 and normalized.isalpha():
        return normalized
    match = re.search(r"\b(?:ANSWER|OPTION)?\s*[:\-]?\s*\(?([A-Z])\)?\b", normalized)
    if match:
        return match.group(1)
    # Prefer a trailing standalone option letter (common after long thinking traces).
    trailing = re.search(r"(?:^|\n)\s*\(?([A-Z])\)?\s*$", normalized)
    if trailing:
        return trailing.group(1)
    if choices:
        lowered = text.lower()
        for key, value in choices.items():
            if isinstance(value, str) and value.strip() and value.lower() in lowered:
                return str(key).upper()
    return None


def should_run_task(
    task: dict[str, Any],
    model: str,
    resume: bool,
    keep_existing_responses: bool,
    rerun_errors: bool,
) -> bool:
    existing_response = str(task.get("model_response", "")).strip()
    if rerun_errors and existing_response.startswith("[ERROR]"):
        return True
    if keep_existing_responses and existing_response:
        return False
    if resume and task.get("model_response_model") == model and existing_response:
        return False
    return True


def clear_model_response(task: dict[str, Any]) -> None:
    for key in (
        "model_response",
        "model_response_model",
        "model_response_created_at",
        "model_predicted_option",
        "model_is_correct",
        "model_random_baseline",
        "model_response_source_file",
        "model_response_question_text",
        "predicted_option",
        "is_correct",
        "random_baseline",
    ):
        task.pop(key, None)
    task["model_response"] = ""


def apply_result(task: dict[str, Any], raw_text: str, model: str) -> None:
    choices = task.get("choices", {})
    question_format = str(task.get("question_format", ""))
    predicted_key = None
    is_correct = None
    random_baseline = None

    if question_format == "MCQ" and isinstance(choices, dict) and choices:
        predicted_key = extract_answer_key(raw_text, choices)
        random_baseline = 1.0 / float(len(choices))
        ground_truth = str(task.get("ground_truth", "")).strip().upper()
        if ground_truth:
            is_correct = predicted_key == ground_truth

    task["model_response"] = raw_text
    task["model_response_model"] = model
    task["model_response_created_at"] = datetime.now().isoformat(timespec="seconds")
    task["model_predicted_option"] = predicted_key
    task["model_is_correct"] = is_correct
    task["model_random_baseline"] = random_baseline


def output_summary(tasks: list[dict[str, Any]], model: str) -> dict[str, Any]:
    model_rows = [task for task in tasks if task.get("model_response_model") == model]
    mcq_rows = [
        task
        for task in model_rows
        if task.get("question_format") == "MCQ"
        and isinstance(task.get("choices"), dict)
        and task.get("choices")
    ]
    correct = sum(1 for task in mcq_rows if task.get("model_is_correct") is True)
    return {
        "model": model,
        "total_tasks": len(tasks),
        "model_response_count": len(model_rows),
        "model_response_by_question_id": dict(sorted(Counter(task.get("id") for task in model_rows).items())),
        "mcq_evaluated_count": len(mcq_rows),
        "mcq_correct_count": correct,
        "mcq_accuracy": correct / len(mcq_rows) if mcq_rows else None,
        "by_question_format": dict(Counter(task.get("question_format") for task in tasks)),
    }


def run_batches(
    tasks: list[dict[str, Any]],
    task_indices: list[int],
    processor: AutoProcessor,
    llm: LLM,
    sampling_params: SamplingParams,
    resolver: VisualResolver,
    args: argparse.Namespace,
    output_payload: dict[str, Any],
    pass_name: str,
) -> None:
    completed_since_save = 0
    total = len(task_indices)
    batch_offsets = range(0, total, args.batch_size)
    progress = None
    if tqdm is not None:
        progress = tqdm(total=total, desc=pass_name, unit="task", dynamic_ncols=True)

    for offset in batch_offsets:
        batch_indices = task_indices[offset : offset + args.batch_size]
        requests: list[dict[str, Any]] = []
        runnable_indices: list[int] = []
        for idx in batch_indices:
            task = tasks[idx]
            try:
                images = resolver.load_images(task)
                prompt = build_prompt(processor, task, num_images=len(images))
                requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
                runnable_indices.append(idx)
            except Exception as exc:
                apply_result(task, f"[ERROR] {exc}", args.model)
                completed_since_save += 1

        if requests:
            outputs = llm.generate(requests, sampling_params=sampling_params)
            for idx, output in zip(runnable_indices, outputs):
                text = output.outputs[0].text.strip() if output.outputs else ""
                apply_result(tasks[idx], text, args.model)
                completed_since_save += 1

        done = min(offset + len(batch_indices), total)
        if progress is not None:
            progress.update(len(batch_indices))
            progress.set_postfix(
                saved_after=f"{args.save_every}",
                warnings=sum(resolver.warnings.values()),
                refresh=False,
            )
        else:
            print(f"{pass_name}: [{done}/{total}] completed current pass")
        if args.save_every > 0 and completed_since_save >= args.save_every:
            output_payload["meta"]["vlm_annotation_summary"] = output_summary(tasks, args.model)
            output_payload["meta"]["vlm_annotation_warnings"] = dict(resolver.warnings)
            write_json(args.output, output_payload)
            completed_since_save = 0

    if progress is not None:
        progress.close()


def main() -> None:
    args = parse_args()
    args.hf_home = resolve_hf_home(args.model, args.hf_home, args.local_files_only)
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if "CUDA_VISIBLE_DEVICEs" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError(
            "Found CUDA_VISIBLE_DEVICEs, but CUDA_VISIBLE_DEVICES is not set. "
            "Environment variables are case-sensitive; use CUDA_VISIBLE_DEVICES=2."
        )

    input_payload = read_json(args.tasks)
    output_payload = sanitize_payload(input_payload)

    if args.resume and args.output.exists():
        output_payload = sanitize_payload(merge_resume_payload(output_payload, read_json(args.output)))

    all_tasks = output_payload.get("tasks", [])
    requested_ids = {str(value) for value in args.only_question_id if str(value).strip()}
    if args.start_index:
        candidate_indices = list(range(args.start_index, len(all_tasks)))
    else:
        candidate_indices = list(range(len(all_tasks)))
    if requested_ids:
        candidate_indices = [idx for idx in candidate_indices if str(all_tasks[idx].get("id", "")) in requested_ids]
    if args.max_tasks > 0:
        candidate_indices = candidate_indices[: args.max_tasks]

    if not args.keep_existing_responses and not args.resume:
        for idx in candidate_indices:
            clear_model_response(all_tasks[idx])
    elif requested_ids and args.resume:
        for idx in candidate_indices:
            if str(all_tasks[idx].get("id", "")) in requested_ids:
                clear_model_response(all_tasks[idx])

    output_payload.setdefault("meta", {})
    has_sc6 = any(is_sc6_task(all_tasks[idx]) for idx in candidate_indices)
    effective_max_model_len = max(args.max_model_len, SC6_MAX_MODEL_LEN) if has_sc6 else args.max_model_len
    effective_batch_size = min(args.batch_size, 2) if has_sc6 else args.batch_size
    output_payload["meta"]["vlm_annotation_config"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "hf_home": str(args.hf_home.resolve()),
        "local_files_only": args.local_files_only,
        "formatted_scenes_dir": str(args.formatted_scenes_dir.resolve()),
        "nuscenes_root": str(args.nuscenes_root.resolve()),
        "batch_size": effective_batch_size,
        "max_model_len": effective_max_model_len,
        "mcq_max_tokens": args.mcq_max_tokens,
        "oeq_max_tokens": args.oeq_max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "object_reference_prompt_value": MASKED_OBJECT_REFERENCE,
        "sc6_frame_policy": f"all_{SC6_FRAME_COUNT}_scene_grid_frames" if has_sc6 else None,
        "only_question_id": sorted(requested_ids) if requested_ids else None,
    }

    if args.sanitize_only:
        output_payload["meta"]["vlm_annotation_summary"] = output_summary(all_tasks, args.model)
        write_json(args.output, output_payload)
        print(f"Wrote sanitized JSON to: {args.output}")
        return

    runnable = [
        idx
        for idx in candidate_indices
        if should_run_task(
            all_tasks[idx],
            model=args.model,
            resume=args.resume,
            keep_existing_responses=args.keep_existing_responses,
            rerun_errors=args.rerun_errors,
        )
    ]
    mcq_indices = [idx for idx in runnable if all_tasks[idx].get("question_format") == "MCQ"]
    oeq_indices = [idx for idx in runnable if all_tasks[idx].get("question_format") != "MCQ"]
    print(
        f"Loaded {len(all_tasks)} tasks; running {len(runnable)} "
        f"({len(mcq_indices)} MCQ, {len(oeq_indices)} OEQ)."
    )

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        args.model,
        cache_dir=str(args.hf_home / "hub"),
        local_files_only=args.local_files_only,
    )
    llm = LLM(
        model=args.model,
        download_dir=str(args.hf_home / "hub"),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=effective_max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    mcq_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.mcq_max_tokens,
        repetition_penalty=1.0,
    )
    oeq_sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.oeq_max_tokens,
        repetition_penalty=1.0,
    )
    resolver = VisualResolver(
        formatted_scenes_dir=args.formatted_scenes_dir,
        nuscenes_root=args.nuscenes_root,
        version=args.version,
        image_cache_size=args.image_cache_size,
        render_missing_boxes=args.render_missing_boxes,
    )
    args.batch_size = effective_batch_size
    try:
        if mcq_indices:
            print("Running MCQ pass...")
            run_batches(
                all_tasks,
                mcq_indices,
                processor,
                llm,
                mcq_sampling,
                resolver,
                args,
                output_payload,
                "MCQ annotation",
            )
        if oeq_indices:
            print("Running OEQ pass...")
            run_batches(
                all_tasks,
                oeq_indices,
                processor,
                llm,
                oeq_sampling,
                resolver,
                args,
                output_payload,
                "OEQ annotation",
            )
    finally:
        resolver.close()

    output_payload["meta"]["vlm_annotation_summary"] = output_summary(all_tasks, args.model)
    output_payload["meta"]["vlm_annotation_warnings"] = dict(resolver.warnings)
    write_json(args.output, output_payload)
    print(f"Saved annotated JSON to: {args.output}")


if __name__ == "__main__":
    main()
