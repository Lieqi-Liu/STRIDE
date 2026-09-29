# Reference evaluation runners

These scripts are copied from the internal `Spatial_Temporal_Intelligence` workspace.
They retain lab-specific defaults (GPU layouts, absolute data roots, `MODEL_SPECS` catalogs).

**Preferred public API:** use the `stride` package (`python -m stride.cli score ...`).

Use this folder when you need to reproduce the exact research runners (vLLM VLM batches,
Gemini/Claude/OpenAI experts, Dolphins, vision-judge batches). Before running:

1. Set `STRIDE_NUSCENES_FORMATTED_SCENES` / `STRIDE_WAYMO_IMAGE_ROOT` (or edit hardcoded paths).
2. Point `--input-json` at `data/nuscenes/stride_nuscenes_v6.json` or `data/waymo/stride_waymo_v1.json`.
3. Install the matching heavy deps (`vllm`, Vertex ADC, Dolphins env, etc.).

| File | Role |
|------|------|
| `score_miniset_oeq_*.py` | BLEURT / trajectory / vision-judge scorers |
| `run_*_miniset*.py` | Inference runners |
| `vlm_batch_utils.py` | Shared MCQ parse + VisualResolver |
| `refresh_waymo_summary.py` | Rebuild Waymo leaderboard CSV |
