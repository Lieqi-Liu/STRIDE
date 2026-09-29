from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def strip_thinking(text: str) -> str:
    """Strip Qwen/R1-style reasoning traces before the final answer."""
    marker = "</" + "think>"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text.strip()
