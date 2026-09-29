"""Vision helpers for resolving nuScenes / Waymo frames at eval time."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def env_path(name: str, default: str | None = None) -> Path | None:
    value = os.environ.get(name, default)
    return Path(value).expanduser() if value else None


def waymo_image_root() -> Path:
    root = env_path("STRIDE_WAYMO_IMAGE_ROOT")
    if root is None:
        raise EnvironmentError(
            "Set STRIDE_WAYMO_IMAGE_ROOT to the Waymo val/images directory "
            "(the folder that contains <scene_id>/CAM_FRONT/...)."
        )
    return root


def nuscenes_formatted_scenes_dir() -> Path:
    root = env_path("STRIDE_NUSCENES_FORMATTED_SCENES")
    if root is None:
        raise EnvironmentError(
            "Set STRIDE_NUSCENES_FORMATTED_SCENES to the formatted_scenes directory "
            "containing per-scene multi-camera grid frames."
        )
    return root


def resolve_waymo_paths(task: dict[str, Any], image_root: Path | None = None) -> list[Path]:
    root = image_root or waymo_image_root()
    paths = task.get("image_paths") or []
    if not paths and task.get("image_path"):
        paths = [task["image_path"]]
    resolved: list[Path] = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        resolved.append(path)
    return resolved


def draw_red_bbox(image: Image.Image, bbox_xyxy: list[float] | None) -> Image.Image:
    if not bbox_xyxy or len(bbox_xyxy) < 4:
        return image
    out = image.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    for width in range(3):
        draw.rectangle([x1 - width, y1 - width, x2 + width, y2 + width], outline=(255, 0, 0))
    return out


def load_waymo_images(
    task: dict[str, Any],
    *,
    image_root: Path | None = None,
    overlay_bbox: bool = True,
) -> list[Image.Image]:
    paths = resolve_waymo_paths(task, image_root=image_root)
    images: list[Image.Image] = []
    bbox = task.get("bbox_xyxy") if overlay_bbox else None
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        if overlay_bbox and bbox is not None and i == len(paths) - 1:
            img = draw_red_bbox(img, bbox)
        images.append(img)
    return images


def resolve_nuscenes_group_file(
    task: dict[str, Any],
    formatted_scenes_dir: Path | None = None,
) -> Path:
    root = formatted_scenes_dir or nuscenes_formatted_scenes_dir()
    rel = task.get("source_group_file")
    if not rel:
        scene_id = task.get("scene_id")
        group_id = str(task.get("group_id", "")).split("_obj")[0]
        rel = f"{scene_id}/{group_id}_vehicle_annotations.json"
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing nuScenes group file: {path}")
    return path


def load_nuscenes_grid_paths(
    task: dict[str, Any],
    formatted_scenes_dir: Path | None = None,
) -> list[Path]:
    """Resolve the 5 chronological multi-camera grid frames for a nuScenes task."""
    root = formatted_scenes_dir or nuscenes_formatted_scenes_dir()
    qid = str(task.get("id", ""))
    scene_id = str(task.get("scene_id", ""))
    scene_dir = root / scene_id

    if qid == "SC-6":
        # Full-scene overview: use evenly spaced grids if many exist.
        grids = sorted(scene_dir.glob("*_grid.jpg"))
        if not grids:
            raise FileNotFoundError(f"No grid frames in {scene_dir}")
        if len(grids) <= 8:
            return grids
        # Even subsample to 8 frames.
        idxs = [round(i * (len(grids) - 1) / 7) for i in range(8)]
        return [grids[i] for i in idxs]

    group_file = resolve_nuscenes_group_file(task, root)
    with group_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    frame_indices = payload.get("frame_indices_1based", [])
    if len(frame_indices) != 5:
        raise ValueError(f"Expected 5 frame indices in {group_file}, got {len(frame_indices)}")

    image_paths: list[Path] = []
    for idx in frame_indices:
        matches = sorted(scene_dir.glob(f"{idx:03d}_*_grid.jpg"))
        if not matches:
            raise FileNotFoundError(f"Missing frame image for index {idx} in {scene_dir}")
        image_paths.append(matches[0])

    # Prefer pre-rendered selected-vehicle overlay on the query frame when present.
    base_group = payload.get("group_id") or str(task.get("group_id", "")).split("_obj")[0]
    selected_path = scene_dir / f"{base_group}_selected_vehicle_render.jpg"
    if selected_path.exists():
        image_paths[4] = selected_path
    return image_paths


def load_nuscenes_images(
    task: dict[str, Any],
    *,
    formatted_scenes_dir: Path | None = None,
) -> list[Image.Image]:
    return [
        Image.open(path).convert("RGB")
        for path in load_nuscenes_grid_paths(task, formatted_scenes_dir=formatted_scenes_dir)
    ]
