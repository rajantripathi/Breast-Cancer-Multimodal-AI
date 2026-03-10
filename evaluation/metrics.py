from __future__ import annotations

"""Metric helpers for enterprise-style evaluation."""

from collections import Counter
import random
from typing import Iterable, Sequence


def label_distribution(labels: Iterable[str]) -> dict[str, int]:
    """Count labels in an iterable."""
    return dict(Counter(labels))


def expected_calibration_error(
    y_true: Sequence[int],
    confidences: Sequence[float],
    predictions: Sequence[int],
    num_bins: int = 10,
) -> float:
    """Compute expected calibration error with pure Python bins."""
    if not y_true:
        return 0.0
    ece = 0.0
    for bin_index in range(num_bins):
        start = bin_index / num_bins
        end = (bin_index + 1) / num_bins
        members = [
            index
            for index, confidence in enumerate(confidences)
            if confidence >= start and (confidence < end or (bin_index == num_bins - 1 and confidence <= end))
        ]
        if not members:
            continue
        accuracy = sum(int(predictions[index] == y_true[index]) for index in members) / len(members)
        avg_confidence = sum(confidences[index] for index in members) / len(members)
        ece += (len(members) / len(y_true)) * abs(accuracy - avg_confidence)
    return float(ece)


def bootstrap_metric(
    metric_fn,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[Sequence[float]] | None = None,
    iterations: int = 1000,
    seed: int = 7,
) -> tuple[float, float]:
    """Compute a 95 percent bootstrap confidence interval."""
    rng = random.Random(seed)
    values: list[float] = []
    indices = list(range(len(y_true)))
    for _ in range(iterations):
        sample_idx = [rng.choice(indices) for _ in indices]
        sample_true = [y_true[index] for index in sample_idx]
        sample_pred = [y_pred[index] for index in sample_idx]
        sample_score = [y_score[index] for index in sample_idx] if y_score is not None else None
        try:
            value = metric_fn(sample_true, sample_score if sample_score is not None else sample_pred)
        except Exception:
            continue
        if value == value:
            values.append(float(value))
    if not values:
        return 0.0, 0.0
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return lower, upper
