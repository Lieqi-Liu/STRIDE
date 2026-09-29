# STRIDE Evaluation

## Data Preparation

1. Follow [Data Preparation](../README.md#data-preparation) to download upstream data and build STRIDE media, then **verify**:

   ```bash
   python scripts/prepare_nuscenes_media.py \
     --nuscenes-root /path/to/nuscenes/trainval \
     --output-dir /path/to/formatted_scenes
   export STRIDE_NUSCENES_FORMATTED_SCENES=/path/to/formatted_scenes
   python scripts/verify_stride_media.py --split nuscenes --require-overlays

   python scripts/prepare_waymo_media.py \
     --waymo-image-root /path/to/waymo/val/images
   export STRIDE_WAYMO_IMAGE_ROOT=/path/to/waymo/val/images
   python scripts/verify_stride_media.py --split waymo
   ```

2. Run inference with your LVLM / expert stack and save results as a single JSON file with a top-level `tasks` list. Each task must include identifying fields (`id`, `scene_id`, `group_id`, …) plus `model_response`.

   You can start from a blank template:

   ```bash
   python evaluation/make_prediction_template.py \
     --split nuscenes \
     --output $ROOT_TO_RESULTS/my_model_responses.json
   ```

   Or use the bundled OpenAI-compatible runner (loads media via `stride.visual`):

   ```bash
   python evaluation/run_openai.py \
     --split waymo --limit 5 \
     --model gpt-4.1 \
     --output $ROOT_TO_RESULTS/smoke_responses.json
   ```

3. If you are using a subset of STRIDE (e.g., the Mini demo) for a smoke test, point `--split` to `nuscenes_mini` / `waymo_mini`. Reporting numbers on Mini is **not** valid for the leaderboard — use the full `nuscenes` / `waymo` splits.

4. Now the data organization will be like:

```
├── STRIDE
│   ├── nuScenes
│   │   └── questions.json
│   ├── Waymo
│   │   └── questions.json
│   └── Mini
├── $STRIDE_NUSCENES_FORMATTED_SCENES   # grids + selected_vehicle overlays
├── $STRIDE_WAYMO_IMAGE_ROOT            # Waymo val/images
├── $ROOT_TO_RESULTS
│   └── my_model_responses.json
```



## Instructions

1. Install dependencies for evaluation via pip.
  ```bash
   pip install -e .
   # optional: BLEURT open-OEQ scoring
   pip install -e ".[bleurt]"
   # optional: OpenAI runner
   pip install -e ".[openai]"
  ```
2. Score MCQ + trajectory metrics (CPU-friendly). By default only tasks with a non-empty `model_response` are scored; use `--no-score-answered-only` for official full-split scoring where missing answers count as incorrect.
  ```bash
   python -m stride.cli score \
     --split nuscenes \
     --predictions $ROOT_TO_RESULTS/my_model_responses.json \
     --metrics mcq,trajectory \
     --no-score-answered-only \
     --output-dir eval_out/my_model \
     --write-scored-predictions
  ```
3. (Optional) Score open-ended OEQs with BLEURT.
  ```bash
   python -m stride.cli score \
     --split nuscenes \
     --predictions $ROOT_TO_RESULTS/my_model_responses.json \
     --metrics mcq,trajectory,bleurt \
     --no-score-answered-only \
     --output-dir eval_out/my_model
  ```
4. (Optional) Run the vision LLM judge for open OEQs (requires vLLM + Qwen3-VL-32B + image roots). See `stride/vision_judge.py` and `evaluation/reference/score_miniset_oeq_llm_judge.py`.
5. Smoke-test with the bundled sample predictions:
  ```bash
   python -m stride.cli score \
     --split nuscenes \
     --predictions evaluation/examples/sample_predictions_nuscenes.json \
     --metrics mcq,trajectory \
     --output-dir eval_out/sample
  ```



## Metrics


| Metric                   | Definition                                          | Direction |
| ------------------------ | --------------------------------------------------- | --------- |
| **MCQ accuracy**         | `#correct / #MCQ` (unparsed → incorrect)            | higher    |
| **BLEURT**               | `Elron/bleurt-base-512` hyp vs GT on open OEQs      | higher    |
| **Vision judge**         | Qwen3-VL-32B scores 0–10 with frames (optional)     | higher    |
| **ADE / FDE / mini-FDE** | Waypoint L2 errors; invalid format → stay-at-origin | lower     |


Default open-OEQ id sets:

- nuScenes: `SC-6, SP-7, SU-7, TE-6, TRJ-7`
- Waymo: `SC-4, SP-3a, TRJ-7, TRJ-9`

Trajectory ids: `TRJ-5` (5 waypoints), `TRJ-6` (4 waypoints).

Official ranking sorts primarily by **MCQ accuracy**. There is no fused overall scalar.

## Prediction format

```
{
  "meta": {"model": "my-model", "split": "nuscenes"},
  "tasks": [
    {
      "id": "SP-1",
      "scene_id": "scene_012",
      "group_id": "group_002",
      "object_id": "...",
      "question_format": "MCQ",
      "model_response": "B"
    }
  ]
}
```



## Leaderboard files

Published baseline tables live in `[leaderboard/](./leaderboard)`. To propose an update, open an issue/PR with your `*_score_report.json` and model details.

## Reference runners

The `[reference/](./reference)` folder mirrors internal research runners (vLLM VLMs, Gemini/Claude/OpenAI experts, Dolphins, vision-judge batches). They retain lab-specific defaults; prefer `python -m stride.cli` for public evaluation.