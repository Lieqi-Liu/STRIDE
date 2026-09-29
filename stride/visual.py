"""Vision helpers for resolving nuScenes / Waymo frames at eval time."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_TRAJECTORY_NAME = re.compile(r"Candidate trajectory ([A-Z])\b", re.IGNORECASE)
_TRAJECTORY_COLORS = {
    "A": (255, 64, 64),
    "B": (64, 160, 255),
    "C": (64, 220, 120),
    "D": (255, 180, 64),
    "E": (200, 120, 255),
}


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
    stroke = max(3, int(0.006 * max(out.size)))
    for width in range(stroke):
        draw.rectangle([x1 - width, y1 - width, x2 + width, y2 + width], outline=(255, 0, 0))
    return out


def trajectory_letter_map(task: dict[str, Any]) -> dict[str, str]:
    """Map stored trajectory keys to the option letter shown on the image."""
    candidates = task.get("candidate_trajectories") or task.get("candidate_trajectories_image")
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


def _trajectory_font(image_height: int) -> ImageFont.ImageFont:
    size = max(18, int(round(image_height * 0.034)))
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_trajectory_inset(
    candidates: dict[str, Any],
    letters: dict[str, str],
    *,
    panel_px: int,
) -> Image.Image:
    """Top-down ego diagram (up = forward, left = left). No calibration needed."""
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
    max_x = max(xs) if xs else 1.0
    max_abs_y = max((abs(v) for v in ys), default=1.0)
    span = max(max_x, 2.0 * max_abs_y, 1.0)
    usable = panel_px - 2 * margin

    def to_px(x: float, y: float) -> tuple[float, float]:
        # vehicle x forward -> image up; vehicle y left -> image left
        px = margin + usable * 0.5 - (y / span) * usable
        py = margin + usable - (x / span) * usable
        return px, py

    # axes
    origin = to_px(0.0, 0.0)
    draw.line([origin, to_px(span * 0.9, 0.0)], fill=(180, 180, 180, 200), width=2)
    draw.line([to_px(0.0, -span * 0.4), to_px(0.0, span * 0.4)], fill=(120, 120, 120, 160), width=1)
    ego_r = max(4, panel_px // 40)
    draw.ellipse(
        (origin[0] - ego_r, origin[1] - ego_r, origin[0] + ego_r, origin[1] + ego_r),
        fill=(240, 240, 240, 255),
        outline=(0, 0, 0, 255),
    )

    for letter, path, _reach in sorted(prepared, key=lambda row: row[2], reverse=True):
        color = _TRAJECTORY_COLORS.get(letter, (255, 255, 255))
        pts = [to_px(p[0], p[1]) for p in path]
        flat = [c for p in pts for c in p]
        if len(pts) >= 2:
            draw.line(flat, fill=color + (255,), width=max(3, panel_px // 80))
        ex, ey = pts[-1]
        r = max(10, panel_px // 18)
        draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=color + (255,), outline=(255, 255, 255, 255))
        draw.text((ex, ey), letter, fill=(0, 0, 0, 255), font=font, anchor="mm")
    return panel


def draw_candidate_trajectories(image: Image.Image, task: dict[str, Any]) -> Image.Image:
    """Overlay TRJ ranking options using preprojected pixels + bird's-eye inset."""
    candidates_m = task.get("candidate_trajectories")
    candidates_px = task.get("candidate_trajectories_image")
    if not isinstance(candidates_m, dict) and not isinstance(candidates_px, dict):
        return image
    if not candidates_m and not candidates_px:
        return image

    letters = trajectory_letter_map(task)
    painted = image.copy()
    width, height = painted.size
    draw = ImageDraw.Draw(painted)
    font = _trajectory_font(height)
    stroke = max(4, int(round(height * 0.008)))
    radius = max(14, int(round(height * 0.022)))

    # Prefer preprojected image polylines (no calibration needed).
    if isinstance(candidates_px, dict) and candidates_px:
        rendered: list[tuple[str, list[tuple[float, float]], float]] = []
        for name, points in candidates_px.items():
            if not isinstance(points, list) or len(points) < 2:
                continue
            polyline = [(float(p[0]), float(p[1])) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
            if len(polyline) < 2:
                continue
            letter = letters.get(str(name), str(name))
            # Approximate reach from ego-frame if available.
            meters = (candidates_m or {}).get(name) if isinstance(candidates_m, dict) else None
            reach = float(meters[-1][0]) if isinstance(meters, list) and meters else float(len(polyline))
            rendered.append((letter, polyline, reach))
        for letter, polyline, _reach in sorted(rendered, key=lambda row: row[2], reverse=True):
            color = _TRAJECTORY_COLORS.get(letter, (255, 255, 255))
            flat = [c for p in polyline for c in p]
            draw.line(flat, fill=(255, 255, 255), width=stroke + 4)
            draw.line(flat, fill=color, width=stroke)
            end_x, end_y = polyline[-1]
            end_x = min(max(end_x, radius + 2), width - radius - 2)
            end_y = min(max(end_y, radius + 2), height - radius - 2)
            draw.ellipse(
                (end_x - radius, end_y - radius, end_x + radius, end_y + radius),
                fill=color,
                outline=(255, 255, 255),
                width=3,
            )
            draw.text((end_x, end_y), letter, fill=(0, 0, 0), font=font, anchor="mm")

    if isinstance(candidates_m, dict) and candidates_m:
        inset = _draw_trajectory_inset(
            candidates_m,
            letters,
            panel_px=max(220, int(round(width * 0.30))),
        )
        painted.paste(inset, (12, 12), inset)
    return painted


def load_waymo_images(
    task: dict[str, Any],
    *,
    image_root: Path | None = None,
    overlay_bbox: bool = True,
    overlay_trajectories: bool = True,
) -> list[Image.Image]:
    paths = resolve_waymo_paths(task, image_root=image_root)
    if not paths:
        raise FileNotFoundError(f"No Waymo image_paths on task {task.get('id')}")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Waymo images (set STRIDE_WAYMO_IMAGE_ROOT): {missing[0]}")
    images: list[Image.Image] = []
    bbox = task.get("bbox_xyxy") if overlay_bbox else None
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        if i == len(paths) - 1:
            if overlay_bbox and bbox is not None:
                img = draw_red_bbox(img, bbox)
            if overlay_trajectories and (
                task.get("candidate_trajectories") or task.get("candidate_trajectories_image")
            ):
                img = draw_candidate_trajectories(img, task)
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
        grids = sorted(scene_dir.glob("*_grid.jpg"))
        if not grids:
            raise FileNotFoundError(f"No grid frames in {scene_dir}")
        if len(grids) <= 8:
            return grids
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
