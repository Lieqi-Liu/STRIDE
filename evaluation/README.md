# STRIDE Evaluation

## Data Preparation

1. First of all, follow the instructions on [Data Preparation](../README.md#data-preparation) to obtain STRIDE annotations and image roots. Set:

   ```bash
   export STRIDE_NUSCENES_FORMATTED_SCENES=/path/to/formatted_scenes
   export STRIDE_WAYMO_IMAGE_ROOT=/path/to/waymo/val/images
   ```

2. Run inference with your LVLM / expert stack and save results as a single JSON file with a top-level `tasks` list. Each task must include identifying fields (`id`, `scene_id`, `group_id`, …) plus `model_response`.

   You can start from a blank template:

   ```bash
   python evaluation/make_prediction_template.py \
     --split nuscenes_v6 \
     --output $ROOT_TO_RESULTS/my_model_responses.json
   ```

3. If you are using a subset of STRIDE (e.g., the Mini demo) for a smoke test, point `--split` to `nuscenes_mini` / `waymo_mini`. Reporting numbers on Mini is **not** valid for the leaderboard — use the full `nuscenes_v6` / `waymo_v1` splits.

4. Now the data organization will be like:

```
├── STRIDE
│   ├── nuScenes
│   │   └── questions.json
│   ├── Waymo
│   │   └── questions.json
│   └── Mini
├── $ROOT_TO_RESULTS
│   └── my_model_responses.json
```

## Instructions

0. Install dependencies for evaluation via pip.

   ```bash
   pip install -e .
   # optional: BLEURT open-FRQ scoring
   pip install -e ".[bleurt]"
   # optional: OpenAI runner
   pip install -e ".[openai]"
   ```

1. Score MCQ + trajectory metrics (CPU-friendly). By default only tasks with a non-empty `model_response` are scored; use `--no-score-answered-only` for official full-split scoring where missing answers count as incorrect.

   ```bash
   python -m stride.cli score \
     --split nuscenes_v6 \
     --predictions $ROOT_TO_RESULTS/my_model_responses.json \
     --metrics mcq,trajectory \
     --no-score-answered-only \
     --output-dir eval_out/my_model \
     --write-scored-predictions
   ```

2. (Optional) Score open-ended FRQs with BLEURT.

   ```bash
   python -m stride.cli score \
     --split nuscenes_v6 \
     --predictions $ROOT_TO_RESULTS/my_model_responses.json \
     --metrics mcq,trajectory,bleurt \
     --no-score-answered-only \
     --output-dir eval_out/my_model
   ```

3. (Optional) Run the vision LLM judge for open FRQs (requires vLLM + Qwen3-VL-32B + image roots). See `stride/vision_judge.py` and `evaluation/reference/score_miniset_frq_llm_judge.py`.

4. Smoke-test with the bundled sample predictions:

   ```bash
   python -m stride.cli score \
     --split nuscenes_v6 \
     --predictions evaluation/examples/sample_predictions_nuscenes.json \
     --metrics mcq,trajectory \
     --output-dir eval_out/sample
   ```

## Metrics

| Metric | Definition | Direction |
| ------ | ---------- | --------- |
| **MCQ accuracy** | `#correct / #MCQ` (unparsed → incorrect) | higher |
| **BLEURT** | `Elron/bleurt-base-512` hyp vs GT on open FRQs | higher |
| **Vision judge** | Qwen3-VL-32B scores 0–10 with frames (optional) | higher |
| **ADE / FDE / mini-FDE** | Waypoint L2 errors; invalid format → stay-at-origin | lower |

Default open-FRQ id sets:

- nuScenes: `SC-6, SP-7, SU-7, TE-6, TRJ-7`
- Waymo: `SC-4, SP-3a, TRJ-7, TRJ-9`

Trajectory ids: `TRJ-5` (5 waypoints), `TRJ-6` (4 waypoints).

Official ranking sorts primarily by **MCQ accuracy**. There is no fused overall scalar.

## Prediction format

```
{
  "meta": {"model": "my-model", "split": "nuscenes_v6"},
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

Published baseline tables live in [`leaderboard/`](./leaderboard). To propose an update, open an issue/PR with your `*_score_report.json` and model details.

## Reference runners

The [`reference/`](./reference) folder mirrors internal research runners (vLLM VLMs, Gemini/Claude/OpenAI experts, Dolphins, vision-judge batches). They retain lab-specific defaults; prefer `python -m stride.cli` for public evaluation.
