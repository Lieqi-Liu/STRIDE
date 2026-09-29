# scripts/

| Script | Purpose |
|--------|---------|
| `prepare_nuscenes_media.py` | Stitch 2×3 grids, install group sidecars, render selected-vehicle query overlays (`nuscenes-devkit` required for overlays) |
| `prepare_waymo_media.py` | Verify Waymo `val/images` paths + smoke-load a few tasks (no SAM; boxes/trajectories are in `questions.json`) |
| `verify_stride_media.py` | Load every task through `stride.visual` so missing media fails before inference |

Typical flow after downloading upstream data:

```bash
python scripts/prepare_nuscenes_media.py --nuscenes-root ... --output-dir ...
export STRIDE_NUSCENES_FORMATTED_SCENES=...
python scripts/verify_stride_media.py --split nuscenes --require-overlays

python scripts/prepare_waymo_media.py --waymo-image-root ...
export STRIDE_WAYMO_IMAGE_ROOT=...
python scripts/verify_stride_media.py --split waymo
```
