#!/usr/bin/env python3
"""Verify Waymo CAM_FRONT images referenced by STRIDE are present.

Waymo needs no stitching and no SAM/bbox preprocessing at user time:
questions already store relative paths under val/images/, plus precomputed
`bbox_xyxy` and `candidate_trajectories_image` in questions.json.
Eval overlays the red box / TRJ ranking curves from those fields.

Example:

    python scripts/prepare_waymo_media.py \\
      --waymo-image-root /path/to/waymo/val/images

Then:

    export STRIDE_WAYMO_IMAGE_ROOT=/path/to/waymo/val/images
    python scripts/verify_stride_media.py --split waymo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "STRIDE" / "Waymo" / "questions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--waymo-image-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--max-missing-print", type=int, default=20)
    parser.add_argument("--load-smoke", type=int, default=5, help="Also load N tasks through stride.visual (0 disables).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.waymo_image_root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Waymo image root not found: {root}")
    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    paths: list[str] = []
    for task in payload.get("tasks", []):
        for key in ("image_paths", "history_image_paths"):
            for p in task.get(key) or []:
                paths.append(str(p))
        for key in ("image_path", "anchor_image_path"):
            if task.get(key):
                paths.append(str(task[key]))
    unique = list(dict.fromkeys(paths))
    missing = [p for p in unique if not (root / p).exists()]
    print(f"[waymo] root={root}")
    print(f"[waymo] referenced={len(unique)} missing={len(missing)}")
    for p in missing[: args.max_missing_print]:
        print(f"  missing: {p}")
    if missing:
        raise SystemExit(1)

    if args.load_smoke > 0:
        sys.path.insert(0, str(ROOT))
        from stride.visual import load_waymo_images

        os.environ["STRIDE_WAYMO_IMAGE_ROOT"] = str(root)
        picked: list[dict] = []
        for pred in (
            lambda t: bool(t.get("bbox_xyxy")),
            lambda t: bool(t.get("candidate_trajectories")),
            lambda t: True,
        ):
            for t in payload.get("tasks", []):
                if pred(t) and t not in picked:
                    picked.append(t)
                    break
            if len(picked) >= args.load_smoke:
                break
        for task in picked[: args.load_smoke]:
            images = load_waymo_images(task)
            print(f"[waymo] smoke-load {task.get('id')} frames={len(images)} size={images[0].size}")
            for im in images:
                im.close()

    print(f"[done] export STRIDE_WAYMO_IMAGE_ROOT={root}")
    print("[done] next: python scripts/verify_stride_media.py --split waymo")


if __name__ == "__main__":
    main()
