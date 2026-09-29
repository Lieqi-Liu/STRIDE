#!/usr/bin/env python3
"""Run VLMs on the Waymo miniset split (separate from the nuScenes split).

Reuses nuScenes model loading / MCQ scoring, but loads Waymo CAM_FRONT frames
directly and overlays red boxes from bbox_xyxy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
NUSCENES_DIR = SCRIPT_DIR.parent / "full_nuscenes"
if str(NUSCENES_DIR) not in sys.path:
    sys.path.insert(0, str(NUSCENES_DIR))

import run_miniset_model_benchmark as nusc_bench  # noqa: E402

ANNOTATOR = nusc_bench.ANNOTATOR
MASKED_OBJECT_REFERENCE = ANNOTATOR.MASKED_OBJECT_REFERENCE
MAX_IMAGE_PIXELS = ANNOTATOR.MAX_IMAGE_PIXELS

DEFAULT_INPUT_JSON = SCRIPT_DIR / "questions_with_answers_waymo_miniset.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "miniset_model_benchmark_outputs"
DEFAULT_HF_HOME = Path("/local1/lieqiliu/huggingface")
DEFAULT_CAMERA_CALIBRATION_DIR = Path(
    "/local1/rgao727/waymo_dataset/val/annotations/camera_calibration"
)
_TRAJECTORY_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
_TRAJECTORY_NAME = re.compile(r"candidate trajectory ([A-Z])\s*$", re.IGNORECASE)
_TRAJECTORY_COLORS = {
    "A": (220, 38, 38),
    "B": (0, 148, 210),
    "C": (230, 170, 0),
}

MODEL_SPECS = [
    {
        "name": "qwen2_5_vl_3b",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "backend": "vllm",
        "enabled": True,
        "batch_size": 8,
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
        "batch_size": 4,
        "local_files_only": False,
        "enable_thinking": False,
        "note": "Qwen3.6 multimodal MoE; requires vLLM>=0.19.",
    },
    {
        "name": "qwen3_6_27b",
        "model_id": "Qwen/Qwen3.6-27B",
        "backend": "vllm",
        "enabled": True,
        "batch_size": 4,
        "local_files_only": False,
        "enable_thinking": False,
        "note": "Qwen3.6 multimodal dense; requires vLLM>=0.19.",
    },
    {
        "name": "qwen3_6_35b_a3b_fp8",
        "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
        "backend": "vllm",
        "enabled": True,
        "batch_size": 4,
        "local_files_only": False,
        "enable_thinking": False,
        "note": "FP8 Qwen3.6 MoE for 1.5-GPU packing; thinking disabled.",
    },
    {
        "name": "qwen3_6_27b_fp8",
        "model_id": "Qwen/Qwen3.6-27B-FP8",
        "backend": "vllm",
        "enabled": True,
        "batch_size": 4,
        "local_files_only": False,
        "enable_thinking": False,
        "note": "FP8 Qwen3.6 dense for 1.5-GPU packing; thinking disabled.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLMs on the Waymo miniset split and summarize MCQ outcomes."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable vLLM CUDA graphs (default: true).",
    )
    parser.add_argument("--mcq-max-tokens", type=int, default=8)
    parser.add_argument("--oeq-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--image-cache-size", type=int, default=64)
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
    )
    parser.add_argument(
        "--only-question-id",
        action="append",
        default=[],
        help="Run only tasks whose id matches (e.g. TRJ-5). Can be repeated.",
    )
    parser.add_argument(
        "--only-question-format",
        action="append",
        default=[],
        choices=("MCQ", "OEQ", "mcq", "oeq"),
    )
    parser.add_argument(
        "--resume-from-output",
        action="store_true",
        help="Merge existing per-model response JSON and only rerun selected question ids.",
    )
    parser.add_argument(
        "--draw-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay bbox_xyxy as a red box on the query frame.",
    )
    return parser.parse_args()


def waymo_image_paths(task: dict[str, Any]) -> list[Path]:
    raw_paths = task.get("image_paths") or task.get("history_image_paths") or []
    paths = [Path(str(path)) for path in raw_paths if str(path).strip()]
    query = task.get("anchor_image_path") or task.get("image_path")
    if query:
        query_path = Path(str(query))
        if not paths or paths[-1] != query_path:
            paths.append(query_path)
    if not paths:
        raise FileNotFoundError(
            f"No Waymo image paths for {task.get('question_id') or task.get('id')}"
        )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Waymo image: {missing[0]}")
    return paths


def draw_red_bbox(image: Image.Image, bbox_xyxy: list[Any]) -> Image.Image:
    if len(bbox_xyxy) != 4:
        return image
    painted = image.copy()
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    width, height = painted.size
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    stroke = max(4, int(0.006 * max(width, height)))
    draw = ImageDraw.Draw(painted)
    for inset in range(stroke):
        draw.rectangle((x1 - inset, y1 - inset, x2 + inset, y2 + inset), outline=(255, 0, 0))
    return painted


def trajectory_letter_map(task: dict[str, Any]) -> dict[str, str]:
    """Map a stored trajectory name to the option letter drawn on the image.

    Option rebalancing swaps choice text without moving the waypoint arrays, so
    the letter the model must answer can differ from the key in
    ``candidate_trajectories``.
    """
    candidates = task.get("candidate_trajectories")
    if not isinstance(candidates, dict) or not candidates:
        return {}
    identity = {str(name): str(name) for name in candidates}
    choices = task.get("choices")
    if not isinstance(choices, dict) or not choices:
        return identity
    renamed: dict[str, str] = {}
    for option, text in choices.items():
        match = _TRAJECTORY_NAME.search(str(text).strip())
        if match is None:
            return identity
        renamed[match.group(1).upper()] = str(option)
    if set(renamed) != set(identity):
        return identity
    return renamed


def aligned_mcq_choices(task: dict[str, Any]) -> dict[str, str] | None:
    """Choice text whose letter matches the label drawn on that path."""
    choices = task.get("choices")
    if not isinstance(choices, dict) or not choices or not trajectory_letter_map(task):
        return None
    mismatched = False
    for option, text in choices.items():
        match = _TRAJECTORY_NAME.search(str(text).strip())
        if match is None:
            return None
        if match.group(1).upper() != str(option):
            mismatched = True
    if not mismatched:
        return None
    return {str(option): f"Candidate trajectory {option}" for option in choices}


def candidate_overlay_note(task: dict[str, Any]) -> str:
    letters = sorted(set(trajectory_letter_map(task).values()))
    if not letters:
        return ""
    shown = ", ".join(letters)
    return (
        "Candidate ego trajectories are drawn on the query frame and labeled "
        f"{shown}. The same trajectories are also shown in a top-down diagram "
        "in the top-left corner, where up is forward and left is left.\n"
    )


def apply_trajectory_prompt(task: dict[str, Any], prompt: str) -> str:
    note = candidate_overlay_note(task)
    if not note:
        return prompt
    aligned = aligned_mcq_choices(task)
    if aligned is not None:
        prompt = re.sub(
            r"Choices: .*\n",
            "Choices: " + json.dumps(aligned, ensure_ascii=False) + "\n",
            prompt,
            count=1,
        )
    marker = "a red bounding box.\n"
    if marker in prompt:
        return prompt.replace(marker, marker + note, 1)
    return note + prompt


def _query_frame_name(task: dict[str, Any], query_path: Path) -> str:
    frame_name = str(task.get("frame_name") or "").strip()
    if frame_name:
        return frame_name
    stem = query_path.stem
    suffix = "_FRONT"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def _clip_code(x: float, y: float, width: int, height: int) -> int:
    code = 0
    if x < 0:
        code |= 1
    elif x > width - 1:
        code |= 2
    if y < 0:
        code |= 8
    elif y > height - 1:
        code |= 4
    return code


def _clip_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x0, y0 = start
    x1, y1 = end
    code0 = _clip_code(x0, y0, width, height)
    code1 = _clip_code(x1, y1, width, height)
    for _ in range(8):
        if code0 == 0 and code1 == 0:
            return (x0, y0), (x1, y1)
        if code0 & code1:
            return None
        code = code0 or code1
        if code & 8:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (0 - y0) / (y1 - y0)
            y = 0.0
        elif code & 4:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * ((height - 1) - y0) / (y1 - y0)
            y = float(height - 1)
        elif code & 2:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * ((width - 1) - x0) / (x1 - x0)
            x = float(width - 1)
        else:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (0 - x0) / (x1 - x0)
            x = 0.0
        if code == code0:
            x0, y0 = x, y
            code0 = _clip_code(x0, y0, width, height)
        else:
            x1, y1 = x, y
            code1 = _clip_code(x1, y1, width, height)
    return None


def _clip_polyline(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    clipped: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        segment = _clip_segment(start, end, width, height)
        if segment is None:
            continue
        if not clipped or abs(clipped[-1][0] - segment[0][0]) > 0.5 or abs(clipped[-1][1] - segment[0][1]) > 0.5:
            clipped.append(segment[0])
        clipped.append(segment[1])
    if len(points) == 1:
        x, y = points[0]
        if 0 <= x < width and 0 <= y < height:
            clipped.append((x, y))
    return clipped


def _trajectory_font(image_height: int) -> ImageFont.ImageFont:
    size = max(22, int(round(image_height * 0.034)))
    try:
        return ImageFont.truetype(str(_TRAJECTORY_FONT), size)
    except OSError:
        return ImageFont.load_default()


def draw_candidate_trajectories(
    image: Image.Image,
    task: dict[str, Any],
    *,
    calibration_dir: Path = DEFAULT_CAMERA_CALIBRATION_DIR,
) -> Image.Image:
    candidates = task.get("candidate_trajectories")
    if not isinstance(candidates, dict) or not candidates:
        return image
    query_path = Path(str(task.get("anchor_image_path") or task.get("image_path") or ""))
    frame_name = _query_frame_name(task, query_path)
    calib_path = calibration_dir / f"{frame_name}.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing camera calibration: {calib_path}")
    payload = json.loads(calib_path.read_text())
    front = next(
        (
            calib
            for calib in payload.get("camera_calibrations", [])
            if str(calib.get("name", "")).upper() == "FRONT"
        ),
        None,
    )
    if front is None:
        raise RuntimeError(f"FRONT calibration missing in {calib_path}")
    intrinsic = front.get("intrinsic") or []
    extrinsic = front.get("extrinsic") or []
    if len(intrinsic) < 4 or len(extrinsic) != 16:
        raise RuntimeError(f"Incomplete FRONT calibration in {calib_path}")

    painted = image.copy()
    width, height = painted.size
    calib_w = float(front.get("width") or width)
    calib_h = float(front.get("height") or height)
    fx = float(intrinsic[0]) * width / calib_w
    fy = float(intrinsic[1]) * height / calib_h
    cx = float(intrinsic[2]) * width / calib_w
    cy = float(intrinsic[3]) * height / calib_h
    vehicle_to_cam = np.linalg.inv(np.array(extrinsic, dtype=np.float64).reshape(4, 4))
    letters = trajectory_letter_map(task)

    def project(point: list[Any]) -> tuple[float, float] | None:
        vehicle = np.array([float(point[0]), float(point[1]), 0.0, 1.0], dtype=np.float64)
        camera = vehicle_to_cam @ vehicle
        if camera[0] <= 0.2:
            return None
        return (
            float(cx - fx * (camera[1] / camera[0])),
            float(cy - fy * (camera[2] / camera[0])),
        )

    draw = ImageDraw.Draw(painted)
    rendered: list[tuple[str, list[tuple[float, float]], float]] = []
    for name, points in candidates.items():
        if not isinstance(points, list) or len(points) < 2:
            continue
        projected = [pixel for pixel in (project(point) for point in points) if pixel is not None]
        polyline = _clip_polyline(projected, width, height)
        if len(polyline) < 2:
            continue
        letter = letters.get(str(name), str(name))
        reach = max(float(point[0]) for point in points)
        rendered.append((letter, polyline, reach))

    font = _trajectory_font(height)
    radius = max(16, int(round(height * 0.022)))
    stroke = max(6, int(round(height * 0.008)))
    labels: list[tuple[float, float, str, tuple[int, int, int]]] = []
    for letter, polyline, _reach in sorted(rendered, key=lambda row: row[2], reverse=True):
        color = _TRAJECTORY_COLORS.get(letter, (255, 255, 255))
        flat = [coord for point in polyline for coord in point]
        draw.line(flat, fill=(255, 255, 255), width=stroke + 4)
        draw.line(flat, fill=color, width=stroke)
        end_x, end_y = polyline[-1]
        labels.append((end_x, end_y, letter, color))

    occupied: list[tuple[float, float]] = []
    for end_x, end_y, letter, color in sorted(labels, key=lambda row: row[1]):
        label_x, label_y = end_x, max(radius + 2, end_y - radius * 0.15)
        for used_x, used_y in occupied:
            if (label_x - used_x) ** 2 + (label_y - used_y) ** 2 < (radius * 2.4) ** 2:
                label_x = min(width - radius - 2, used_x + radius * 2.5)
        label_x = min(max(label_x, radius + 2), width - radius - 2)
        label_y = min(max(label_y, radius + 2), height - radius - 2)
        occupied.append((label_x, label_y))
        draw.ellipse(
            (label_x - radius, label_y - radius, label_x + radius, label_y + radius),
            fill=color,
            outline=(255, 255, 255),
            width=3,
        )
        draw.text((label_x, label_y), letter, fill=(0, 0, 0), font=font, anchor="mm")

    inset = _trajectory_inset(candidates, letters, panel_px=max(240, int(round(width * 0.32))))
    painted.paste(inset, (12, 12), inset)
    return painted


def _trajectory_inset(
    candidates: dict[str, Any],
    letters: dict[str, str],
    *,
    panel_px: int,
) -> Image.Image:
    """Top-down ego diagram. Up is forward, left is left."""
    panel = Image.new("RGBA", (panel_px, panel_px), (16, 18, 22, 228))
    draw = ImageDraw.Draw(panel)
    margin = max(18, int(round(panel_px * 0.08)))
    font = _trajectory_font(panel_px)
    xs = [0.0]
    ys = [0.0]
    prepared: list[tuple[str, list[list[float]], float]] = []
    for name, points in candidates.items():
        if not isinstance(points, list) or not points:
            continue
        path = [[0.0, 0.0]] + [[float(point[0]), float(point[1])] for point in points]
        xs.extend(point[0] for point in path)
        ys.extend(point[1] for point in path)
        letter = letters.get(str(name), str(name))
        reach = max(point[0] for point in path)
        prepared.append((letter, path, reach))
    max_x = max(xs)
    max_abs_y = max(abs(value) for value in ys)
    usable = panel_px - 2 * margin
    scale = min(
        usable / max(max_x, 1.0),
        (usable / 2) / max(max_abs_y, 0.5),
    )
    origin_x = panel_px / 2
    origin_y = panel_px - margin

    def to_px(point: list[float]) -> tuple[float, float]:
        return (origin_x - point[1] * scale, origin_y - point[0] * scale)

    draw.polygon(
        [
            (origin_x, origin_y - 16),
            (origin_x - 8, origin_y + 6),
            (origin_x + 8, origin_y + 6),
        ],
        fill=(255, 255, 255, 255),
    )
    radius = max(12, int(round(panel_px * 0.055)))
    occupied: list[tuple[float, float]] = []
    for letter, path, _reach in sorted(prepared, key=lambda row: row[2]):
        color = _TRAJECTORY_COLORS.get(letter, (255, 255, 255))
        pixels = [to_px(point) for point in path]
        flat = [coord for point in pixels for coord in point]
        draw.line(flat, fill=(255, 255, 255, 255), width=7)
        draw.line(flat, fill=color + (255,), width=4)
        end_x, end_y = pixels[-1]
        for used_x, used_y in occupied:
            if (end_x - used_x) ** 2 + (end_y - used_y) ** 2 < (radius * 2.3) ** 2:
                end_x = used_x + radius * 2.2
        end_x = min(max(end_x, radius + 2), panel_px - radius - 2)
        end_y = min(max(end_y, radius + 2), panel_px - radius - 2)
        occupied.append((end_x, end_y))
        draw.ellipse(
            (end_x - radius, end_y - radius, end_x + radius, end_y + radius),
            fill=color + (255,),
            outline=(255, 255, 255, 255),
            width=2,
        )
        draw.text((end_x, end_y), letter, fill=(0, 0, 0, 255), font=font, anchor="mm")
    return panel


def trajectory_cache_token(task: dict[str, Any]) -> tuple[Any, ...] | None:
    candidates = task.get("candidate_trajectories")
    if not isinstance(candidates, dict) or not candidates:
        return None
    letters = trajectory_letter_map(task)
    rows = []
    for name, points in candidates.items():
        flat = tuple(round(float(value), 3) for point in points for value in point[:2])
        rows.append((str(name), letters.get(str(name), str(name)), flat))
    return tuple(rows)


class WaymoVisualResolver:
    def __init__(
        self,
        *,
        image_cache_size: int,
        draw_boxes: bool,
        calibration_dir: Path = DEFAULT_CAMERA_CALIBRATION_DIR,
    ) -> None:
        self.image_cache_size = image_cache_size
        self.draw_boxes = draw_boxes
        self.calibration_dir = calibration_dir
        self.image_cache: OrderedDict[tuple[Any, ...], list[Image.Image]] = OrderedDict()
        self.warnings: dict[str, int] = defaultdict(int)

    def close(self) -> None:
        for images in self.image_cache.values():
            for image in images:
                image.close()
        self.image_cache.clear()

    def cache_key(self, task: dict[str, Any], paths: list[Path]) -> tuple[Any, ...]:
        bbox = task.get("bbox_xyxy") if self.draw_boxes else None
        bbox_key = tuple(float(v) for v in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None
        return (
            tuple(str(path) for path in paths),
            bbox_key,
            task.get("object_id"),
            trajectory_cache_token(task),
        )

    def load_images(self, task: dict[str, Any]) -> list[Image.Image]:
        paths = waymo_image_paths(task)
        key = self.cache_key(task, paths)
        cached = self.image_cache.get(key)
        if cached is not None:
            self.image_cache.move_to_end(key)
            return cached

        images: list[Image.Image] = []
        bbox = task.get("bbox_xyxy") if self.draw_boxes else None
        for index, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            if index == len(paths) - 1 and isinstance(bbox, list) and len(bbox) == 4:
                image = draw_red_bbox(image, bbox)
            if index == len(paths) - 1 and task.get("candidate_trajectories"):
                image = draw_candidate_trajectories(
                    image,
                    task,
                    calibration_dir=self.calibration_dir,
                )
            images.append(ANNOTATOR.resize_for_vlm(image, max_pixels=MAX_IMAGE_PIXELS))

        self.image_cache[key] = images
        self.image_cache.move_to_end(key)
        while len(self.image_cache) > self.image_cache_size:
            _, old_images = self.image_cache.popitem(last=False)
            for image in old_images:
                image.close()
        return images


def build_waymo_prompt(
    processor: Any,
    task: dict[str, Any],
    *,
    num_images: int,
    enable_thinking: bool | None = None,
) -> str:
    question = str(task.get("question", "")).replace("<obj>", MASKED_OBJECT_REFERENCE)
    ref = str(task.get("object_reference_original") or task.get("object_reference") or "").strip()
    if ref and ref != MASKED_OBJECT_REFERENCE and ref in question:
        question = question.replace(ref, MASKED_OBJECT_REFERENCE)
    choices = aligned_mcq_choices(task) or task.get("choices", {})
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

    context_seconds = (task.get("hidden_metadata") or {}).get("approx_context_window_seconds")
    window = f" sampled over approximately {context_seconds} seconds" if context_seconds else ""
    context_line = (
        f"You are given {num_images} chronological front-view driving frames{window}. "
        f"The first {max(num_images - 1, 0)} images are temporal context. "
        f"The {num_images}th image is the query frame; when a target object exists, it is highlighted by "
        "a red bounding box.\n"
        f"{candidate_overlay_note(task)}"
    )
    target_line = ""
    if task.get("object_id") or task.get("bbox_xyxy"):
        target_line = f"Target object: {MASKED_OBJECT_REFERENCE} in the {num_images}th image.\n"

    prompt_text = (
        f"{context_line}"
        f"Question ID: {task.get('id', '')}\n"
        f"{target_line}"
        f"Question: {question}\n"
        f"{choice_text}"
        f"{answer_instruction}"
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(num_images)]
            + [{"type": "text", "text": prompt_text}],
        }
    ]
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


def run_indices_vllm(
    tasks: list[dict[str, Any]],
    indices: list[int],
    loaded: nusc_bench.LoadedModel,
    sampling_params: Any,
    resolver: WaymoVisualResolver,
    model_id: str,
    output_path: Path,
    payload: dict[str, Any],
    args: argparse.Namespace,
    desc: str,
) -> int:
    completed_since_save = 0
    completed = 0
    for batch_indices in nusc_bench.iter_batches(indices, args.batch_size, desc):
        requests: list[dict[str, Any]] = []
        runnable_indices: list[int] = []
        for idx in batch_indices:
            try:
                images = resolver.load_images(tasks[idx])
                prompt = build_waymo_prompt(
                    loaded.processor,
                    tasks[idx],
                    num_images=len(images),
                    enable_thinking=getattr(args, "enable_thinking", None),
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
            payload["meta"]["partial_summary"] = nusc_bench.summarize_model(tasks)
            nusc_bench.write_json(output_path, payload)
            completed_since_save = 0
    return completed


def selected_model_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = set(args.only_model)
    specs = []
    for spec in reversed(MODEL_SPECS):
        if not spec.get("enabled", True):
            continue
        if requested and spec["name"] not in requested:
            continue
        specs.append(spec)
    return specs


def run_model_spec(
    model_spec: dict[str, Any],
    input_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = str(model_spec["name"])
    model_id = str(model_spec["model_id"])
    output_path = args.output_dir / f"{nusc_bench.safe_slug(name)}_responses.json"
    started_at = datetime.now().isoformat(timespec="seconds")
    payload, runnable_tasks = nusc_bench.prepare_payload(
        input_payload, args, output_path=output_path
    )
    tasks = payload["tasks"]
    payload["meta"]["benchmark_model"] = model_spec
    payload["meta"]["benchmark_started_at"] = started_at
    payload["meta"]["dataset_split"] = "waymo_miniset"
    payload["meta"]["visual_backend"] = "waymo_cam_front_paths"

    run_args = argparse.Namespace(**vars(args))
    if model_spec.get("batch_size"):
        run_args.batch_size = int(model_spec["batch_size"])
    if "enable_thinking" in model_spec:
        run_args.enable_thinking = bool(model_spec["enable_thinking"])
    use_local_only = nusc_bench.model_local_files_only(model_spec, run_args)

    loaded: nusc_bench.LoadedModel | None = None
    try:
        if not use_local_only:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        loaded = nusc_bench.load_model(model_spec, run_args, local_files_only=use_local_only)
    except Exception as exc:
        return {
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
    finally:
        if args.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        if loaded.backend != "vllm":
            raise ValueError(f"Waymo runner currently supports vLLM only, got {loaded.backend}")
        resolver = WaymoVisualResolver(
            image_cache_size=args.image_cache_size,
            draw_boxes=args.draw_boxes,
        )
        try:
            requested_ids = {str(value) for value in args.only_question_id if str(value).strip()}
            requested_formats = nusc_bench.requested_question_formats(run_args)
            mcq_indices = [
                idx
                for idx, task in enumerate(runnable_tasks)
                if nusc_bench.is_mcq(task)
                and nusc_bench.matches_question_filter(task, requested_ids)
                and nusc_bench.matches_format_filter(task, requested_formats)
            ]
            oeq_indices = [
                idx
                for idx, task in enumerate(runnable_tasks)
                if nusc_bench.is_oeq(task)
                and nusc_bench.matches_question_filter(task, requested_ids)
                and nusc_bench.matches_format_filter(task, requested_formats)
            ]
            run_indices_vllm(
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
            run_indices_vllm(
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

        nusc_bench.merge_runnable_results(tasks, runnable_tasks)
        model_summary = nusc_bench.summarize_model(tasks)
        payload["meta"]["benchmark_finished_at"] = datetime.now().isoformat(timespec="seconds")
        payload["meta"]["benchmark_summary"] = model_summary
        nusc_bench.write_json(output_path, payload)
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
        nusc_bench.write_json(output_path, payload)
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
            "partial_summary": nusc_bench.summarize_model(tasks),
        }
    finally:
        nusc_bench.cleanup_loaded_model(loaded)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.hf_home = args.hf_home.expanduser()
    nusc_bench.apply_hf_home_env(args.hf_home)
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    input_payload = nusc_bench.read_json(args.input_json)
    if "tasks" not in input_payload and isinstance(input_payload.get("questions"), list):
        raise SystemExit(
            f"{args.input_json} still uses a 'questions' list. "
            "Run build_waymo_miniset_50.py first so the split uses {'meta','tasks'}."
        )

    specs = selected_model_specs(args)
    if not specs:
        raise SystemExit("No matching enabled MODEL_SPECS. Check --only-model.")
    results = []
    for spec in specs:
        print(f"\n=== Waymo split: {spec['name']} ({spec['model_id']}) ===", flush=True)
        result = run_model_spec(spec, input_payload, args)
        results.append(result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in {"load_traceback", "inference_traceback"}},
                indent=2,
            ),
            flush=True,
        )

    summary_path = args.output_dir / "benchmark_summary.json"
    enabled_specs = [spec for spec in MODEL_SPECS if spec.get("enabled", True)]
    enabled_model_names = {str(spec["name"]) for spec in enabled_specs}
    merged_results = nusc_bench.merge_existing_results(summary_path, results, enabled_model_names)
    existing_by_name = {str(result.get("name", "")): result for result in merged_results}
    # build_benchmark_summary looks up MODEL_SPECS in the nuScenes module; patch locally.
    original_specs = nusc_bench.MODEL_SPECS
    nusc_bench.MODEL_SPECS = MODEL_SPECS
    try:
        benchmark_summary = nusc_bench.build_benchmark_summary(
            input_json=args.input_json,
            output_dir=args.output_dir,
            model_specs=enabled_specs,
            last_run_model_specs=specs,
            existing_results_by_name=existing_by_name,
        )
    finally:
        nusc_bench.MODEL_SPECS = original_specs
    benchmark_summary["dataset_split"] = "waymo_miniset"
    nusc_bench.write_json(summary_path, benchmark_summary)
    print(f"\nWrote Waymo benchmark summary: {summary_path}")


if __name__ == "__main__":
    main()
