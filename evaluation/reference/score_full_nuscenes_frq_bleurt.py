#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_INPUT = Path("/local1/lieqiliu/nuscenes/fullset/questions_with_answers_all_qwen3vl30b.json")
DEFAULT_OUTPUT_DIR = Path("/local1/lieqiliu/nuscenes/fullset/frq_bleurt_scores_qwen3vl30b")
DEFAULT_BLEURT_MODEL = "Elron/bleurt-base-512"
DEFAULT_HF_HOME = Path("/local1/lieqiliu/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score full NuScenes FRQ Qwen responses against ground truth with BLEURT."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bleurt-model", default=DEFAULT_BLEURT_MODEL)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only locally cached HF model files. Use --no-local-files-only to download.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--task-id", action="append", default=[], help="Optional FRQ id filter.")
    parser.add_argument("--max-rows", type=int, default=0, help="If >0, score only first N rows.")
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--write-back", action="store_true")
    parser.add_argument(
        "--write-back-output",
        type=Path,
        default=None,
        help="If --write-back, write updated JSON here instead of overwriting --input.",
    )
    return parser.parse_args()


def model_cache_dir_name(model: str) -> str:
    return f"models--{model.replace('/', '--')}"


def resolve_hf_home(model: str, requested_hf_home: Path, local_files_only: bool) -> Path:
    requested_hf_home = requested_hf_home.expanduser()
    requested_model_dir = requested_hf_home / "hub" / model_cache_dir_name(model)
    if requested_model_dir.exists():
        return requested_hf_home

    default_hf_home = Path.home() / ".cache" / "huggingface"
    default_model_dir = default_hf_home / "hub" / model_cache_dir_name(model)
    if default_model_dir.exists():
        print(
            f"[cache] {model} not found under {requested_hf_home}; "
            f"using cached model under {default_hf_home}."
        )
        return default_hf_home

    if local_files_only:
        raise FileNotFoundError(
            f"Local model cache not found for {model}. Checked:\n"
            f"  - {requested_model_dir}\n"
            f"  - {default_model_dir}\n"
            "Run again with --no-local-files-only if you want to download it."
        )
    return requested_hf_home


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def is_frq(row: dict[str, Any]) -> bool:
    return str(row.get("question_format", "")).upper() == "FRQ"


def collect_rows(
    tasks: list[dict[str, Any]], task_ids: set[str], include_errors: bool, max_rows: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for index, task in enumerate(tasks):
        if not is_frq(task):
            counters["non_frq"] += 1
            continue
        if task_ids and str(task.get("id", "")) not in task_ids:
            counters["filtered_task_id"] += 1
            continue
        gt = str(task.get("ground_truth", "")).strip()
        response = str(task.get("model_response", "")).strip()
        if not gt:
            counters["missing_ground_truth"] += 1
            continue
        if not response:
            counters["missing_model_response"] += 1
            continue
        if response.startswith("[ERROR]") and not include_errors:
            counters["error_model_response"] += 1
            continue
        row = dict(task)
        row["_task_index"] = index
        rows.append(row)
        counters["scored_candidates"] += 1
        if max_rows > 0 and len(rows) >= max_rows:
            counters["max_rows_stop"] += 1
            break
    return rows, counters


def bleurt_scores_batch(
    hypotheses: list[str],
    references: list[str],
    model_name: str,
    hf_home: Path,
    local_files_only: bool,
    batch_size: int,
    device: str,
) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(hf_home / "hub"),
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=str(hf_home / "hub"),
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    scores: list[float] = []
    iterator = range(0, len(hypotheses), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(hypotheses) + batch_size - 1) // batch_size, desc="BLEURT")
    for start in iterator:
        batch_h = hypotheses[start : start + batch_size]
        batch_r = references[start : start + batch_size]
        encoded = tokenizer(
            batch_r,
            batch_h,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
        if logits.ndim == 0:
            scores.append(float(logits.detach().cpu().item()))
        else:
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
        if tqdm is None:
            print(f"[{min(start + batch_size, len(hypotheses))}/{len(hypotheses)}] scored")
    return scores


def bucket(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def summarize(scored_rows: list[dict[str, Any]], skip_counts: Counter[str]) -> dict[str, Any]:
    by_task_id: dict[str, list[float]] = defaultdict(list)
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in scored_rows:
        score = float(row["bleurt_model_gt"])
        by_task_id[str(row.get("id", ""))].append(score)
        by_task[str(row.get("task", ""))].append(score)
    all_scores = [float(row["bleurt_model_gt"]) for row in scored_rows]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scored_count": len(scored_rows),
        "skip_counts": dict(skip_counts),
        "overall": bucket(all_scores),
        "by_question_id": {key: bucket(values) for key, values in sorted(by_task_id.items())},
        "by_task": {key: bucket(values) for key, values in sorted(by_task.items())},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_index",
        "id",
        "task",
        "scene_id",
        "group_id",
        "object_id",
        "bleurt_model_gt",
        "question",
        "ground_truth",
        "model_response",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task_index": row.get("_task_index"),
                    "id": row.get("id"),
                    "task": row.get("task"),
                    "scene_id": row.get("scene_id"),
                    "group_id": row.get("group_id"),
                    "object_id": row.get("object_id"),
                    "bleurt_model_gt": row.get("bleurt_model_gt"),
                    "question": row.get("question"),
                    "ground_truth": row.get("ground_truth"),
                    "model_response": row.get("model_response"),
                }
            )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    args.hf_home = resolve_hf_home(args.bleurt_model, args.hf_home, args.local_files_only)
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    payload = read_json(args.input)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Input JSON must contain a top-level tasks list.")

    rows, skip_counts = collect_rows(tasks, set(args.task_id), args.include_errors, args.max_rows)
    if not rows:
        raise SystemExit("No FRQ rows are ready for BLEURT scoring.")

    device = resolve_device(args.device)
    hypotheses = [str(row.get("model_response", "")).strip() for row in rows]
    references = [str(row.get("ground_truth", "")).strip() for row in rows]
    scores = bleurt_scores_batch(
        hypotheses=hypotheses,
        references=references,
        model_name=args.bleurt_model,
        hf_home=args.hf_home,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
        device=device,
    )
    for row, score in zip(rows, scores):
        row["bleurt_model_gt"] = score
        row["bleurt_model"] = args.bleurt_model

    summary = summarize(rows, skip_counts)
    summary["input"] = str(args.input.resolve())
    summary["bleurt_model"] = args.bleurt_model
    summary["hf_home"] = str(args.hf_home.resolve())
    summary["device"] = device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "frq_bleurt_scores.json", {"meta": summary, "rows": rows})
    write_csv(args.output_dir / "frq_bleurt_scores.csv", rows)

    if args.write_back:
        for row in rows:
            task = tasks[int(row["_task_index"])]
            task["bleurt_model_gt"] = row["bleurt_model_gt"]
            task["bleurt_model"] = args.bleurt_model
            task["bleurt_scored_at"] = summary["created_at"]
        payload.setdefault("meta", {})
        payload["meta"]["frq_bleurt_summary"] = summary
        write_json(args.write_back_output or args.input, payload)

    print(f"Scored FRQ rows: {summary['scored_count']}")
    print(f"Mean BLEURT model-vs-GT: {summary['overall']['mean']}")
    print(f"Wrote: {args.output_dir / 'summary.json'}")
    print(f"Wrote: {args.output_dir / 'frq_bleurt_scores.csv'}")


if __name__ == "__main__":
    main()
