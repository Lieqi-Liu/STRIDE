#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from run_miniset_model_benchmark import (
    DEFAULT_INPUT_JSON,
    DEFAULT_OUTPUT_DIR,
    MODEL_SPECS,
    build_benchmark_summary,
    read_json,
    safe_slug,
    write_json,
)
from score_full_nuscenes_oeq_bleurt import (
    DEFAULT_BLEURT_MODEL,
    DEFAULT_HF_HOME,
    collect_rows,
    bleurt_scores_batch,
    resolve_device,
    resolve_hf_home,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score miniset benchmark OEQ responses with BLEURT and refresh benchmark_summary.json."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bleurt-model", default=DEFAULT_BLEURT_MODEL)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model", action="append", default=[], help="Only score these model names.")
    parser.add_argument(
        "--response-json",
        type=Path,
        default=None,
        help="Score this response JSON directly (name = stem without _responses).",
    )
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--skip-summary-refresh", action="store_true")
    parser.add_argument(
        "--only-question-id",
        action="append",
        default=[],
        help="Only score these OEQ ids (e.g. SC-4). Repeatable. Default: all OEQs.",
    )
    return parser.parse_args()


def score_response_file(
    response_path: Path,
    *,
    bleurt_model: str,
    hf_home: Path,
    local_files_only: bool,
    batch_size: int,
    device: str,
    include_errors: bool,
    question_ids: set[str] | None = None,
) -> dict[str, Any]:
    payload = read_json(response_path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"{response_path} must contain a top-level tasks list.")

    rows, skip_counts = collect_rows(tasks, question_ids or set(), include_errors, max_rows=0)
    if not rows:
        return {
            "response_path": str(response_path),
            "scored_count": 0,
            "skip_counts": dict(skip_counts),
            "overall": {"count": 0, "mean": None, "min": None, "max": None},
        }

    def strip_thinking(text: str) -> str:
        # Qwen Thinking models emit reasoning then the final answer after </think>.
        marker = "</" + "think>"
        if marker in text:
            return text.rsplit(marker, 1)[-1].strip()
        # Fallback: if the model never closed the think block, keep the last paragraph.
        parts = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        return parts[-1] if parts else text.strip()

    hypotheses = [strip_thinking(str(row.get("model_response", ""))) for row in rows]
    references = [str(row.get("ground_truth", "")).strip() for row in rows]
    scores = bleurt_scores_batch(
        hypotheses=hypotheses,
        references=references,
        model_name=bleurt_model,
        hf_home=hf_home,
        local_files_only=local_files_only,
        batch_size=batch_size,
        device=device,
    )
    scored_at = datetime.now().isoformat(timespec="seconds")
    for row, score in zip(rows, scores):
        row["bleurt_model_gt"] = score
        row["bleurt_model"] = bleurt_model

    summary = summarize(rows, skip_counts)
    summary["response_path"] = str(response_path.resolve())
    summary["bleurt_model"] = bleurt_model
    summary["device"] = device

    for row in rows:
        task = tasks[int(row["_task_index"])]
        task["bleurt_model_gt"] = row["bleurt_model_gt"]
        task["bleurt_model"] = bleurt_model
        task["bleurt_scored_at"] = scored_at

    payload.setdefault("meta", {})
    payload["meta"]["oeq_bleurt_summary"] = summary
    write_json(response_path, payload)
    return summary


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

    device = resolve_device(args.device)
    selected_names = set(args.model)
    question_ids = {str(value) for value in args.only_question_id if str(value).strip()}
    model_summaries: list[dict[str, Any]] = []

    targets: list[tuple[str, Path]] = []
    if args.response_json is not None:
        response_path = args.response_json.expanduser().resolve()
        name = response_path.stem.removesuffix("_responses")
        targets.append((name, response_path))
    else:
        for spec in MODEL_SPECS:
            if not spec.get("enabled", True):
                continue
            name = str(spec["name"])
            if selected_names and name not in selected_names:
                continue
            targets.append((name, args.output_dir / f"{safe_slug(name)}_responses.json"))

    for name, response_path in targets:
        if not response_path.exists():
            print(f"[skip] missing response file: {response_path}")
            continue
        print(f"[score] {name} -> {response_path}")
        summary = score_response_file(
            response_path,
            bleurt_model=args.bleurt_model,
            hf_home=args.hf_home,
            local_files_only=args.local_files_only,
            batch_size=args.batch_size,
            device=device,
            include_errors=args.include_errors,
            question_ids=question_ids,
        )
        model_summaries.append({"name": name, **summary})
        print(
            f"  scored={summary.get('scored_count', 0)} "
            f"mean_bleurt={summary.get('overall', {}).get('mean')}"
        )

    if not model_summaries:
        raise SystemExit("No miniset response files were scored.")

    if not args.skip_summary_refresh:
        summary_path = args.output_dir / "benchmark_summary.json"
        existing_by_name: dict[str, dict[str, Any]] = {}
        if summary_path.exists():
            prior = read_json(summary_path)
            for result in prior.get("results", []):
                if isinstance(result, dict) and result.get("name"):
                    existing_by_name[str(result["name"])] = result
        benchmark_summary = build_benchmark_summary(
            input_json=args.input_json,
            output_dir=args.output_dir,
            model_specs=[spec for spec in MODEL_SPECS if spec.get("enabled", True)],
            existing_results_by_name=existing_by_name,
        )
        write_json(summary_path, benchmark_summary)
        print(f"[summary] refreshed {summary_path}")


if __name__ == "__main__":
    main()
