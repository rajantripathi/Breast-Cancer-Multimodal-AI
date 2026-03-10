from __future__ import annotations

from collections import Counter
from typing import Iterable


def label_distribution(labels: Iterable[str]) -> dict[str, int]:
    return dict(Counter(labels))

