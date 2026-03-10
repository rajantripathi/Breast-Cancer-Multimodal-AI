from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))


def flatten_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        parts = []
        for key, value in payload.items():
            parts.append(f"{key} {flatten_payload(value)}")
        return " ".join(parts)
    if isinstance(payload, list):
        return " ".join(flatten_payload(item) for item in payload)
    return str(payload)


def stable_shuffle(values: list[Any], seed: int = 7) -> list[Any]:
    items = list(values)
    random.Random(seed).shuffle(items)
    return items
