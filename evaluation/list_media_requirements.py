#!/usr/bin/env python3
"""List media files referenced by a STRIDE split (for packaging / download checks).

    python evaluation/list_media_requirements.py --split waymo_v1 --output /tmp/waymo_media.txt
    python evaluation/list_media_requirements.py --split nuscenes_v6 --output /tmp/nusc_groups.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stride.aggregate import load_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, tasks = load_tasks(args.split)
    lines: list[str] = []
    if str(args.split).startswith("waymo"):
        seen = set()
        for t in tasks:
            for p in t.get("image_paths") or []:
                if p not in seen:
                    seen.add(p)
                    lines.append(str(p))
            for p in t.get("history_image_paths") or []:
                if p not in seen:
                    seen.add(p)
                    lines.append(str(p))
    else:
        seen = set()
        for t in tasks:
            rel = t.get("source_group_file")
            if rel and rel not in seen:
                seen.add(rel)
                lines.append(str(rel))
            sid = t.get("scene_id")
            if sid:
                lines.append(f"scene:{sid}")

    # unique preserve order for scene lines too
    ordered = list(dict.fromkeys(lines))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ordered) + "\n")
    print(f"Wrote {len(ordered)} entries -> {args.output}")


if __name__ == "__main__":
    main()
