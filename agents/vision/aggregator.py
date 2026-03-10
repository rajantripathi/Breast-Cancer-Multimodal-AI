from __future__ import annotations

"""Embedding aggregation helpers for slide or bag-level vision features."""

from math import sqrt
from typing import Iterable


def aggregate_embeddings(embeddings: Iterable[list[float]], pooling: str = "mean") -> list[float]:
    """Aggregate a sequence of embedding vectors into a single representation.

    Args:
        embeddings: Embedding vectors for one logical sample.
        pooling: Pooling mode; currently `mean` or `l2_mean`.

    Returns:
        A pooled embedding vector. Empty input yields an empty vector.
    """
    vectors = [vector for vector in embeddings if vector]
    if not vectors:
        return []
    width = len(vectors[0])
    pooled = [0.0] * width
    for vector in vectors:
        for index, value in enumerate(vector[:width]):
            pooled[index] += float(value)
    pooled = [value / len(vectors) for value in pooled]
    if pooling == "l2_mean":
        norm = sqrt(sum(value * value for value in pooled)) or 1.0
        return [value / norm for value in pooled]
    return pooled
