from __future__ import annotations

"""Metric helpers for enterprise-style evaluation."""

from collections import Counter
from typing import Iterable, Sequence

from .statistics import bootstrap_confidence_interval, expected_calibration_error


def label_distribution(labels: Iterable[str]) -> dict[str, int]:
    """Count labels in an iterable."""
    return dict(Counter(labels))


def bootstrap_metric(
    metric_fn,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[Sequence[float]] | None = None,
    iterations: int = 1000,
    seed: int = 7,
    stratify: bool = False,
) -> tuple[float, float]:
    """Compute a 95 percent bootstrap confidence interval."""
    if y_score is not None:
        return bootstrap_confidence_interval(
            lambda sample_true, sample_score: metric_fn(sample_true, sample_score),
            y_true,
            y_score,
            iterations=iterations,
            seed=seed,
            stratify=stratify,
        )
    return bootstrap_confidence_interval(
        lambda sample_true, sample_pred: metric_fn(sample_true, sample_pred),
        y_true,
        y_pred,
        iterations=iterations,
        seed=seed,
        stratify=stratify,
    )
