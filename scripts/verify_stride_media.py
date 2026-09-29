#!/usr/bin/env python3
"""Verify that STRIDE media roots can load every task used at eval time.

Run after prepare_*_media.py:

    export STRIDE_NUSCENES_FORMATTED_SCENES=/path/to/formatted_scenes
    export STRIDE_WAYMO_IMAGE_ROOT=/path/to/waymo/val/images

    python scripts/verify_stride_media.py --split nuscenes
    python scripts/verify_stride_media.py --split waymo

This opens the same image paths / overlays that ``evaluation/run_openai.py``
and ``stride.visual`` use during inference. Scoring (``python -m stride.cli score``)
only needs prediction JSON + shipped questions, but inference needs this check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stride.aggregate import load_tasks
from stride.visual import (
    load_nuscenes_grid_paths,
    load_nuscenes_images,
    load_waymo_images,
    resolve_nuscenes_group_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, help="nuscenes | waymo | nuscenes_mini | waymo_mini")
    parser.add_argument("--limit", type=int, default=0, help="Only check the first N tasks (0 = all).")
    parser.add_argument(
        "--require-overlays",
        action="store_true",
        help="Fail if nuScenes selected-vehicle overlays are missing (recommended for official runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, tasks = load_tasks(args.split)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    backend = "waymo" if str(args.split).startswith("waymo") else "nuscenes"
    ok = 0
    failures: list[str] = []
    overlay_missing = 0

    for i, task in enumerate(tasks):
        tid = f"{task.get('id')}:{task.get('scene_id')}:{task.get('group_id')}"
        try:
            if backend == "waymo":
                images = load_waymo_images(task)
            else:
                paths = load_nuscenes_grid_paths(task)
                if args.require_overlays and str(task.get("id")) != "SC-6":
                    group = None
                    try:
                        gpath = resolve_nuscenes_group_file(task)
                        group = json.loads(gpath.read_text(encoding="utf-8"))
                    except Exception:
                        group = None
                    needs_overlay = bool((group or {}).get("selected_vehicle_annotation_token"))
                    last = paths[-1]
                    if needs_overlay and "selected_vehicle_render" not in last.name:
                        overlay_missing += 1
                        raise FileNotFoundError(
                            f"missing selected_vehicle_render for {tid} (got {last.name}); "
                            "re-run prepare_nuscenes_media.py without --skip-overlays"
                        )
                images = load_nuscenes_images(task)
            if not images:
                raise RuntimeError("loaded 0 images")
            for im in images:
                im.close()
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{tid}: {exc}")
        if (i + 1) % 100 == 0:
            print(f"[progress] checked={i + 1}/{len(tasks)} ok={ok} fail={len(failures)}")

    print(f"[verify] split={args.split} checked={len(tasks)} ok={ok} fail={len(failures)}")
    if overlay_missing:
        print(f"[verify] missing_overlays={overlay_missing}")
    for line in failures[:30]:
        print(f"  FAIL {line}")
    if len(failures) > 30:
        print(f"  ... and {len(failures) - 30} more")
    if failures:
        raise SystemExit(1)
    print("[verify] OK — media is ready for inference + scoring")


if __name__ == "__main__":
    main()
