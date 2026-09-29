# STRIDE

[arXiv](https://arxiv.org/) [Web](https://github.com/Lieqi-Liu/STRIDE) [HF](https://huggingface.co/)

This repository contains the implementation of the paper:

> **STRIDE: Spatial-Temporal Reasoning In Driving Environments**   
>
> [PlusLabNLP](https://github.com/PlusLabNLP)  
>  *arXiv, 2026*

*Overview of STRIDE.*

## Update

- **2026.09**: Initial release of STRIDE annotations (nuScenes v6 + Waymo v1), evaluation toolkit, and baseline leaderboard.
- **2026.09**: Mini demonstration subsets released under `STRIDE/Mini`.

## Data Preparation

The instructions for setting up STRIDE are listed as follows:

1. Clone this repository (question annotations ship with the code under `STRIDE/`).
2. Download the **source images** from the upstream datasets and place them following the layout below:
  - **nuScenes**: obtain [nuScenes](https://www.nuscenes.org/) trainval under the official ToU, then prepare / download the STRIDE `formatted_scenes` multi-camera grids for the miniset scenes.
  - **Waymo**: obtain the [Waymo Open Dataset](https://waymo.com/open/) validation CAM_FRONT images under the official terms.


| Split           | Size | Image Source           | Annotation                             | Notes                                   |
| --------------- | ---- | ---------------------- | -------------------------------------- | --------------------------------------- |
| nuScenes        | 1150 | nuScenes trainval      | `[STRIDE/nuScenes](./STRIDE/nuScenes)` | 23 templates × 50; GT-revised open OEQs |
| Waymo           | 1200 | Waymo Open Dataset val | `[STRIDE/Waymo](./STRIDE/Waymo)`       | 24 templates × 50                       |
| Mini (nuScenes) | 3    | same as nuScenes       | `[STRIDE/Mini](./STRIDE/Mini)`         | Schema demo only                        |
| Mini (Waymo)    | 3    | same as Waymo          | `[STRIDE/Mini](./STRIDE/Mini)`         | Schema demo only                        |


Note that:

1. **STRIDE annotations** (questions + ground truth) are included in this repository under `STRIDE/`.
2. **Images are not vendored** here due to upstream licenses and size (~55 GB nuScenes grids for miniset scenes; ~1 GB unique Waymo CAM_FRONT frames referenced by the miniset).
3. **STRIDE Mini** is a tiny subset for schema demonstration, not for reporting scores.

After setup, the data organization is listed as follows:

```
├── STRIDE
│   ├── nuScenes
│   │   └── questions.json          -- 1,150 nuScenes v6 questions + GT
│   ├── Waymo
│   │   └── questions.json          -- 1,200 Waymo v1 questions + GT (relative image paths)
│   ├── Mini
│   │   ├── nuscenes_mini.json      -- tiny schema demo
│   │   └── waymo_mini.json
│   └── statistics                  -- coverage / diversity JSON+CSV
├── $NUSCENES_FORMATTED_SCENES      -- multi-camera grid frames (external)
│   ├── scene_012
│   │   ├── 001_*_grid.jpg
│   │   ├── group_002_vehicle_annotations.json
│   │   └── ...
│   └── ...
└── $WAYMO_IMAGE_ROOT               -- Waymo val/images (external)
    └── <segment_id>/CAM_FRONT/*.jpg
```

Export the image roots before running vision inference / vision judges:

```bash
export STRIDE_NUSCENES_FORMATTED_SCENES=/path/to/formatted_scenes
export STRIDE_WAYMO_IMAGE_ROOT=/path/to/waymo/val/images
```



## Task Hierarchy

STRIDE probes whether models can reason about **space**, **time**, and **ego motion** in real driving clips. Each sample uses a short temporal window (past → present → future context) with multi-view frames at the query timestep; targets are drawn from **object**, **ego**, and **scene** evidence.

Conceptually, questions cover four capability axes (see overview figure):

- **Spatial** — grounding & understanding of geometry / relations (e.g., distance, relative direction, interactions).
- **Temporal** — memory of past frames & extrapolation of near-future events.
- **Planning** — action-oriented and trajectory outcomes for the ego vehicle.
- **Reasoning** — causal / counterfactual explanations beyond surface description.

Concretely, the released miniset is organized into six template families (50 instances each):


| Family                 | Focus                                     | Example templates |
| ---------------------- | ----------------------------------------- | ----------------- |
| Scene context          | Global environment summary                | `SC-*`            |
| Spatial perception     | Local geometry around a red-boxed object  | `SP-*`            |
| Spatial understanding  | Higher-level spatial inference (nuScenes) | `SU-*`            |
| Temporal memory        | What happened earlier in the clip         | `TM-*`            |
| Temporal extrapolation | Likely near-future outcomes               | `TE-*`            |
| Trajectory             | Ranking, language, or waypoint regression | `TRJ-*`           |


**Question formats.** STRIDE mixes **MCQ**, **open-ended (OEQ)** text, and **geometric** waypoint answers (`TRJ-5` / `TRJ-6`). Visual inputs are **5 chronological frames** (nuScenes: 360° multi-camera grids; Waymo: CAM_FRONT). When a target object exists, it is highlighted by a **red box** on the query frame.

## Data Format

The annotation files contain question-answering pairs as follows (fields may vary slightly by split):

```
{
    "meta": {
        "benchmark": "STRIDE",
        "split": "nuscenes_v6",                 -- or waymo_v1
        "n_questions": 1150,
        "...": "..."
    },
    "tasks": [
        {
            "id": "SP-1",                       -- template id
            "task": "space-perception",         -- task family
            "question_format": "MCQ",           -- MCQ or OEQ
            "scoring_type": "mcq",              -- mcq | open_oeq_text | trajectory_waypoints
            "question": <str>,
            "choices": {                        -- MCQ only
                "A": <str>,
                "B": <str>,
                "..."
            },
            "ground_truth": <str>,              -- letter, free text, or waypoint JSON
            "scene_id": <str>,
            "group_id": <str>,
            "object_id": <str>,                 -- when a target object exists
            "source_group_file": <str>,         -- nuScenes: relative group annotation path
            "image_paths": [<str>, ...],        -- Waymo: 5 paths relative to WAYMO_IMAGE_ROOT
            "bbox_xyxy": [<float>, ...]         -- Waymo: red-box on the query frame
        },
        ...
    ]
}
```



## Data Usage



### ✨ Python API

1. Install dependencies:
  ```bash
   pip install -e .
   # optional BLEURT scoring
   pip install -e ".[bleurt]"
  ```
2. Load a split and inspect tasks:

```python
from stride.aggregate import load_tasks

payload, tasks = load_tasks("nuscenes_v6")  # or "waymo_v1", "nuscenes_mini", "waymo_mini"
print(payload["meta"])
print(tasks[0]["id"], tasks[0]["question"])
```



### Prediction Template

To help users run models against STRIDE, we provide helpers under `evaluation/`:

1. Make sure the directory organization follows [Data Preparation](#data-preparation).
2. Build a blank prediction file:
  ```bash
   python evaluation/make_prediction_template.py \
     --split nuscenes_v6 \
     --output runs/my_model_responses.json
  ```
3. Fill each `model_response`:

  | Format        | Expected `model_response`                                                |
  | ------------- | ------------------------------------------------------------------------ |
  | MCQ           | Single letter `A` / `B` / … (also accepts `ANSWER: A`, `{"answer":"A"}`) |
  | OEQ           | Free-form text                                                           |
  | TRJ-5 / TRJ-6 | JSON list of `[x, y]` waypoints (5 or 4 points)                          |

4. Optionally run the OpenAI-compatible multimodal runner:
  ```bash
   export OPENAI_API_KEY=...
   export STRIDE_NUSCENES_FORMATTED_SCENES=/path/to/formatted_scenes
   python evaluation/run_openai.py \
     --split nuscenes_v6 \
     --model gpt-4.1 \
     --limit 10 \
     --output runs/gpt41_responses.json
  ```



## Leaderboard

Primary ranking key: **MCQ accuracy**. Open-ended and trajectory metrics are reported separately (no single fused score). Random-guess MCQ baselines are ≈19.8% (nuScenes) and ≈20.3% (Waymo).

As in the overview figure, STRIDE is substantially harder than prior driving VQA suites: even GPT-6-Astra drops from ~85% on previous benchmarks to **45.5%** MCQ on STRIDE (nuScenes).

### nuScenes v6


| Rank | Model                     | Type                | MCQ ↑     | BLEURT ↑ | Vision judge ↑ | ADE ↓    | mini-FDE ↓ |
| ---- | ------------------------- | ------------------- | --------- | -------- | -------------- | -------- | ---------- |
| 1    | **gpt-6-astra**           | proprietary         | **45.5%** | −0.303   | 6.41           | 0.60     | 0.98       |
| 2    | GPT-5.5                   | proprietary         | 39.3%     | −0.289   | **6.68**       | 0.65     | 0.97       |
| 3    | gemini-3.6-flash          | proprietary         | 38.9%     | −0.238   | 6.50           | 0.69     | 1.01       |
| 4    | gpt-5.6-sol               | proprietary         | 38.5%     | −0.323   | 5.65           | 0.69     | 0.99       |
| 5    | gemini-3.1-pro-preview    | proprietary         | 37.8%     | −0.301   | 4.53           | 1.42     | 1.84       |
| 6    | Qwen3.6 35B-A3B           | VLM                 | 31.6%     | −0.359   | 5.32           | 7.42     | 5.45       |
| 7    | claude-sonnet-5           | proprietary         | 29.4%     | −0.249   | 6.42           | 0.87     | 1.30       |
| 8    | Qwen2.5-VL 3B             | VLM                 | 27.4%     | —        | 3.62           | —        | —          |
| 9    | Qwen3-VL 30B-A3B-Thinking | VLM                 | 26.8%     | −0.321   | 5.45           | 7.12     | 3.42       |
| 10   | Qwen3-VL 8B-Thinking      | VLM                 | 23.4%     | −0.287   | 5.50           | 8.00     | 3.80       |
| 11   | Dolphins                  | expert              | 18.5%     | −0.435   | 3.24           | 8.16     | 3.81       |
| 12   | Qwen3.6 27B               | VLM                 | 17.5%     | −0.402   | 5.99           | 5.81     | 4.06       |
| —    | UniAD 2.0                 | expert (TRJ-5 only) | —         | —        | —              | **0.56** | **0.79**   |




### Waymo v1


| Rank | Model                     | Type        | MCQ ↑     | BLEURT ↑ | Vision judge ↑ | ADE ↓     | mini-FDE ↓ |
| ---- | ------------------------- | ----------- | --------- | -------- | -------------- | --------- | ---------- |
| 1    | **gpt-6-astra**           | proprietary | **41.2%** | −0.517   | 6.77           | 13.64     | 12.03      |
| 2    | gemini-3.1-pro-preview    | proprietary | 39.7%     | −0.558   | 4.03           | 14.22     | 14.21      |
| 3    | claude-sonnet-5           | proprietary | 39.2%     | −0.504   | **7.01**       | 13.16     | 10.27      |
| 4    | gemini-3.6-flash          | proprietary | 38.8%     | −0.288   | 6.20           | 13.53     | 11.37      |
| 5    | GPT-5.5                   | proprietary | 38.7%     | −0.494   | 7.00           | 13.84     | 12.57      |
| 6    | gpt-5.6-sol               | proprietary | 37.7%     | −0.594   | 6.48           | 13.81     | 12.39      |
| 7    | Qwen2.5-VL 3B             | VLM         | 37.4%     | —        | 2.57           | —         | —          |
| 8    | Qwen3.6 35B-A3B           | VLM         | 34.3%     | −0.539   | 5.83           | 18.55     | 8.86       |
| 9    | Qwen3-VL 8B-Thinking      | VLM         | 32.0%     | −0.663   | 4.72           | 22.36     | 19.37      |
| 10   | Qwen3-VL 30B-A3B-Thinking | VLM         | 31.8%     | —        | 4.98           | **11.83** | **6.18**   |
| 11   | Qwen3-VL 8B-Instruct      | VLM         | 30.1%     | −0.685   | 4.96           | 19.64     | 12.65      |
| 12   | Qwen3.6 27B               | VLM         | 28.4%     | −0.496   | 5.87           | 20.37     | 19.25      |
| 13   | Dolphins                  | expert      | 22.2%     | −0.629   | 1.86           | 18.43     | 8.60       |


Machine-readable tables: `[evaluation/leaderboard](./evaluation/leaderboard)`.

# Evaluation

Check [STRIDE Evaluation](./evaluation) for more details.

## Citation

```bibtex
@misc{stride2026,
  title        = {{STRIDE}: Spatial-Temporal Reasoning In Driving Environments},
  author       = {{PlusLabNLP}},
  year         = {2026},
  howpublished = {\url{https://github.com/PlusLabNLP/STRIDE}},
  note         = {Benchmark release}
}
```

