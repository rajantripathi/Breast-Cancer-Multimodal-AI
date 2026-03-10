from __future__ import annotations

import json
from pathlib import Path


def load_sample_cases(sample_dir: str | Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(Path(sample_dir).glob("*.json"))]

