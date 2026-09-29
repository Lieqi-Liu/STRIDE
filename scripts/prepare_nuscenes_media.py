#!/usr/bin/env python3
"""Build STRIDE nuScenes multi-camera grids from a local nuScenes download.

Self-contained media prep (except downloading nuScenes itself):

1. Download nuScenes trainval under the official ToU.
2. Run this script to stitch the 2x3 camera grids used by STRIDE, install
   the shipped group-annotation sidecars, and render selected-vehicle
   query-frame overlays (requires ``nuscenes-devkit``).

Example:

    pip install nuscenes-devkit matplotlib
    python scripts/prepare_nuscenes_media.py \\
      --nuscenes-root /path/to/nuscenes/trainval \\
      --output-dir /path/to/formatted_scenes

Then:

    export STRIDE_NUSCENES_FORMATTED_SCENES=/path/to/formatted_scenes
    python scripts/verify_stride_media.py --split nuscenes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "STRIDE" / "nuScenes" / "questions.json"
DEFAULT_METADATA = ROOT / "STRIDE" / "nuScenes" / "formatted_metadata"

GRID_CHANNELS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nuscenes-root", type=Path, required=True, help="nuScenes dataset root (contains samples/ and v1.0-trainval/).")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes metadata version folder name.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write formatted_scenes/.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--overwrite-grids", action="store_true", help="Re-render grids even if they already exist.")
    parser.add_argument("--overwrite-overlays", action="store_true", help="Re-render selected-vehicle overlays.")
    parser.add_argument("--skip-grids", action="store_true", help="Only install metadata sidecars (no image stitching).")
    parser.add_argument("--skip-overlays", action="store_true", help="Skip selected-vehicle query overlays.")
    return parser.parse_args()

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compose_grid_image(image_paths: list[Path], output_path: Path) -> bool:
    images: list[Image.Image] = []
    try:
        for path in image_paths:
            if not path.exists():
                print(f"[warn] missing camera image: {path}")
                return False
            images.append(Image.open(path).convert("RGB"))
        cell_w = min(im.width for im in images)
        cell_h = min(im.height for im in images)
        resized = [im.resize((cell_w, cell_h), Image.Resampling.BILINEAR) for im in images]
        grid = Image.new("RGB", (3 * cell_w, 2 * cell_h))
        for idx, im in enumerate(resized):
            grid.paste(im, ((idx % 3) * cell_w, (idx // 3) * cell_h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output_path, format="JPEG", quality=95)
        grid.close()
        for im in resized:
            im.close()
        return True
    finally:
        for im in images:
            im.close()


def install_metadata(metadata_dir: Path, output_dir: Path) -> int:
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Missing shipped metadata directory: {metadata_dir}")
    metadata_dir = metadata_dir.resolve()
    output_dir = output_dir.resolve()
    copied = 0
    for src in metadata_dir.rglob("*"):
        if not src.is_file():
            continue
        # Guard against output_dir living under metadata_dir (would recurse forever).
        try:
            src.resolve().relative_to(output_dir)
            continue
        except ValueError:
            pass
        rel = src.relative_to(metadata_dir)
        if rel.name == "miniset_scenes.json":
            continue
        if rel.name.endswith(".json") is False and "scene_" not in str(rel):
            # keep mapping JSON + per-scene sidecars only
            pass
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def index_sample_data(nuscenes_root: Path, version: str) -> tuple[dict[str, dict], dict[str, dict[str, dict]], dict[str, dict]]:
    """Return (by_token, sample_to_views, stem_to_front_record)."""
    sample_data = read_json(nuscenes_root / version / "sample_data.json")
    by_token = {row["token"]: row for row in sample_data}
    sample_to_views: dict[str, dict[str, dict]] = {}
    stem_to_front: dict[str, dict] = {}
    for row in sample_data:
        filename = str(row.get("filename", ""))
        if not filename.endswith(".jpg"):
            continue
        parts = filename.split("/")
        if len(parts) < 3 or not parts[0] in {"samples", "sweeps"}:
            continue
        channel = parts[1]
        sample_token = row.get("sample_token")
        if channel in GRID_CHANNELS and sample_token and filename.startswith("samples/"):
            sample_to_views.setdefault(sample_token, {})[channel] = row
        if channel == "CAM_FRONT" and filename.startswith("samples/CAM_FRONT/"):
            stem_to_front[Path(filename).stem] = row
    return by_token, sample_to_views, stem_to_front


def walk_prev(by_token: dict[str, dict], start_token: str, steps: int) -> list[dict]:
    """Return [older ..., start] with length steps+1 when possible."""
    chain = [by_token[start_token]]
    cur = start_token
    for _ in range(steps):
        prev = by_token[cur].get("prev")
        if not prev or prev not in by_token:
            break
        chain.append(by_token[prev])
        cur = prev
    chain.reverse()
    return chain


def collect_needed_frames(metadata_dir: Path, questions: Path) -> dict[str, set[int]]:
    """scene_id -> set of 1-based frame indices required by STRIDE."""
    payload = read_json(questions)
    needed: dict[str, set[int]] = {}
    for task in payload.get("tasks", []):
        scene_id = str(task.get("scene_id") or "")
        if not scene_id:
            continue
        needed.setdefault(scene_id, set())
        gf = task.get("source_group_file")
        if gf:
            group_path = metadata_dir / gf
            if group_path.exists():
                group = read_json(group_path)
                for idx in group.get("frame_indices_1based", []):
                    needed[scene_id].add(int(idx))
            continue
        # SC-6 style: use all grids later once we know scene length; mark as all
        needed[scene_id].add(-1)
    return needed


def find_anchor_for_scene(metadata_dir: Path, scene_id: Path | str) -> dict[str, Any] | None:
    scene_dir = metadata_dir / str(scene_id)
    if not scene_dir.exists():
        return None
    for group_path in sorted(scene_dir.glob("group_*_vehicle_annotations.json")):
        group = read_json(group_path)
        if group.get("fifth_frame_cam_front_filename") and group.get("frame_indices_1based"):
            return group
    return None


def render_scene_frames(
    *,
    nuscenes_root: Path,
    scene_id: str,
    frame_indices: set[int],
    metadata_dir: Path,
    output_dir: Path,
    by_token: dict[str, dict],
    sample_to_views: dict[str, dict[str, dict]],
    stem_to_front: dict[str, dict],
    overwrite: bool,
) -> tuple[int, int]:
    anchor = find_anchor_for_scene(metadata_dir, scene_id)
    if anchor is None:
        print(f"[warn] no group metadata for {scene_id}; skip grid render")
        return 0, 0

    fifth_file = str(anchor["fifth_frame_cam_front_filename"])
    fifth_stem = Path(fifth_file).stem
    fifth_record = stem_to_front.get(fifth_stem)
    if fifth_record is None:
        # Fallback via sample token
        sample_token = anchor.get("fifth_frame_sample_token")
        views = sample_to_views.get(str(sample_token), {})
        fifth_record = views.get("CAM_FRONT")
    if fifth_record is None:
        print(f"[warn] cannot resolve CAM_FRONT for {scene_id} ({fifth_stem})")
        return 0, 0

    frame_indices_1based = [int(x) for x in anchor["frame_indices_1based"]]
    fifth_idx = frame_indices_1based[-1]
    # Build a local chain covering from min needed index through fifth_idx.
    wanted = {i for i in frame_indices if i > 0}
    if -1 in frame_indices:
        # Need full scene: walk all the way to chain root then forward via next.
        wanted = set()
        # discover full chain length by walking prev to root then next to end
        root = fifth_record
        while root.get("prev") and root["prev"] in by_token:
            root = by_token[root["prev"]]
        chain = [root]
        cur = root["token"]
        while True:
            nxt = by_token[cur].get("next")
            if not nxt or nxt not in by_token:
                break
            # keep only keyframe sample CAM_FRONT jpgs in samples/
            rec = by_token[nxt]
            fname = str(rec.get("filename", ""))
            if fname.startswith("samples/CAM_FRONT/") and fname.endswith(".jpg"):
                chain.append(rec)
            cur = nxt
        index_to_record = {i + 1: rec for i, rec in enumerate(chain)}
        wanted = set(index_to_record.keys())
    else:
        steps_back = fifth_idx - min(wanted | {fifth_idx})
        chain = walk_prev(by_token, fifth_record["token"], steps=max(steps_back, 0))
        # chain ends at fifth_record; map absolute frame indices
        # chain[-1] corresponds to fifth_idx
        index_to_record = {}
        for offset, rec in enumerate(reversed(chain)):
            index_to_record[fifth_idx - offset] = rec

    rendered = 0
    skipped = 0
    scene_out = output_dir / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    for idx in sorted(wanted):
        rec = index_to_record.get(idx)
        if rec is None:
            # try matching by walking from fifth with exact delta
            delta = fifth_idx - idx
            if delta >= 0:
                chain = walk_prev(by_token, fifth_record["token"], steps=delta)
                if len(chain) == delta + 1:
                    rec = chain[0]
        if rec is None:
            print(f"[warn] {scene_id}: cannot resolve frame {idx}")
            skipped += 1
            continue
        sample_token = rec.get("sample_token")
        views = sample_to_views.get(str(sample_token), {})
        missing = [c for c in GRID_CHANNELS if c not in views]
        if missing:
            print(f"[warn] {scene_id} frame {idx}: missing cameras {missing}")
            skipped += 1
            continue
        front_stem = Path(rec["filename"]).stem
        out_path = scene_out / f"{idx:03d}_{front_stem}_grid.jpg"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        image_paths = [nuscenes_root / views[c]["filename"] for c in GRID_CHANNELS]
        if compose_grid_image(image_paths, out_path):
            rendered += 1
        else:
            skipped += 1
    return rendered, skipped


def keep_right_panel(image_path: Path) -> None:
    """Keep only the right half of the nuScenes render_annotation 1x2 figure."""
    with Image.open(image_path) as image:
        width, height = image.size
        cropped = image.crop((width // 2, 0, width, height))
        cropped.save(image_path, format="JPEG", quality=95)


def render_selected_vehicle_overlays(
    *,
    nuscenes_root: Path,
    version: str,
    metadata_dir: Path,
    output_dir: Path,
    questions: Path,
    overwrite: bool,
) -> tuple[int, int, int]:
    """Render ``{group}_selected_vehicle_render.jpg`` for groups used by STRIDE.

    Uses the shipped ``selected_vehicle_annotation_token`` so overlays match the
    benchmark construction. Requires ``nuscenes-devkit`` (+ matplotlib).
    """
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.geometry_utils import BoxVisibility
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Selected-vehicle overlays require nuscenes-devkit and matplotlib.\n"
            "  pip install nuscenes-devkit matplotlib\n"
            "Or pass --skip-overlays (query frames will be plain grids)."
        ) from exc

    payload = read_json(questions)
    needed_groups: set[str] = set()
    for task in payload.get("tasks", []):
        gf = task.get("source_group_file")
        if gf:
            needed_groups.add(str(gf))

    print(f"[overlays] loading NuScenes SDK ({version})...")
    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)
    written = skipped = failed = 0
    for rel in sorted(needed_groups):
        group_path = metadata_dir / rel
        if not group_path.exists():
            print(f"[warn] missing group sidecar: {rel}")
            failed += 1
            continue
        group = read_json(group_path)
        token = group.get("selected_vehicle_annotation_token")
        scene_id = group.get("scene_id")
        group_id = group.get("group_id")
        if not token or not scene_id or not group_id:
            skipped += 1
            continue
        out_path = output_dir / scene_id / f"{group_id}_selected_vehicle_render.jpg"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            nusc.render_annotation(
                token,
                box_vis_level=BoxVisibility.ANY,
                out_path=str(out_path),
            )
            plt.close("all")
            keep_right_panel(out_path)
            written += 1
        except Exception as exc:  # noqa: BLE001 - keep going across groups
            plt.close("all")
            print(f"[warn] overlay failed {scene_id}/{group_id}: {exc}")
            if out_path.exists():
                out_path.unlink()
            failed += 1
    return written, skipped, failed


def main() -> None:
    args = parse_args()
    nuscenes_root = args.nuscenes_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata_dir = args.metadata_dir.expanduser().resolve()
    version_dir = nuscenes_root / args.version
    if not version_dir.exists():
        raise SystemExit(f"nuScenes version folder not found: {version_dir}")
    if not (nuscenes_root / "samples").exists():
        raise SystemExit(f"nuScenes samples/ not found under {nuscenes_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    n_meta = install_metadata(metadata_dir, output_dir)
    print(f"[meta] installed {n_meta} sidecar files into {output_dir}")

    if not args.skip_grids:
        print("[index] loading nuScenes sample_data.json (large)...")
        by_token, sample_to_views, stem_to_front = index_sample_data(nuscenes_root, args.version)
        print(f"[index] {len(stem_to_front)} CAM_FRONT keyframes, {len(sample_to_views)} samples with views")

        needed = collect_needed_frames(metadata_dir, args.questions)
        total_r = total_s = 0
        for scene_id, frames in sorted(needed.items()):
            r, s = render_scene_frames(
                nuscenes_root=nuscenes_root,
                scene_id=scene_id,
                frame_indices=frames,
                metadata_dir=metadata_dir,
                output_dir=output_dir,
                by_token=by_token,
                sample_to_views=sample_to_views,
                stem_to_front=stem_to_front,
                overwrite=args.overwrite_grids,
            )
            total_r += r
            total_s += s
            print(f"[grids] {scene_id}: rendered={r} existed_or_skipped={s}")
        print(f"[grids] done rendered={total_r} skipped={total_s}")
    else:
        print("[grids] skipped")

    if not args.skip_overlays:
        written, skipped, failed = render_selected_vehicle_overlays(
            nuscenes_root=nuscenes_root,
            version=args.version,
            metadata_dir=metadata_dir,
            output_dir=output_dir,
            questions=args.questions,
            overwrite=args.overwrite_overlays,
        )
        print(f"[overlays] written={written} existed_or_skipped={skipped} failed={failed}")
        if failed:
            print("[warn] some selected-vehicle overlays failed; verify with scripts/verify_stride_media.py")
    else:
        print("[overlays] skipped")

    print(f"[done] export STRIDE_NUSCENES_FORMATTED_SCENES={output_dir}")
    print("[done] next: python scripts/verify_stride_media.py --split nuscenes")


if __name__ == "__main__":
    main()
