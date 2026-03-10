from __future__ import annotations

"""Runtime vision agent backed by extracted feature artifacts."""

import hashlib
import json
from pathlib import Path
from typing import Any

from agents.base_agent import AgentPrediction, BaseAgent
from config import load_settings
from data.common import flatten_payload

from .foundation_models import get_embed_dim, get_model_spec, load_model


def _hash_embedding(seed_text: str, embedding_dim: int) -> list[float]:
    """Create a deterministic pseudo-embedding for smoke and offline flows."""
    values: list[float] = []
    cursor = 0
    while len(values) < embedding_dim:
        digest = hashlib.sha256(f"{seed_text}:{cursor}".encode("utf-8")).digest()
        for byte in digest:
            values.append(round(((byte / 255.0) * 2.0) - 1.0, 6))
            if len(values) >= embedding_dim:
                break
        cursor += 1
    return values


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two embedding vectors."""
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(width))
    left_norm = sum(left[index] * left[index] for index in range(width)) ** 0.5
    right_norm = sum(right[index] * right[index] for index in range(width)) ** 0.5
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


class VisionAgent(BaseAgent):
    """Vision agent that exposes encode and explain in addition to predict."""

    modality = "vision"

    def __init__(self, artifact_path: str | None = None, model_key: str | None = None) -> None:
        configured_model = model_key or str(load_settings().extras.get("vision", {}).get("default_model", "uni2"))
        spec = get_model_spec(configured_model)
        super().__init__(spec.hub, ["normal", "benign", "malignant"], artifact_path=artifact_path)
        self.model_key = spec.name
        self.embedding_dim = get_embed_dim(spec.name)
        self.aggregator = "mean"
        self.class_centroids: dict[str, list[float]] = {}
        self.feature_manifest_path = ""
        self.embedding_backend = "local_deterministic_runtime"
        self.loader = load_model
        if self.artifact_path and self.artifact_path.exists():
            artifact = json.loads(self.artifact_path.read_text())
            self.class_centroids = artifact.get("class_centroids", {})
            self.feature_manifest_path = artifact.get("feature_manifest_path", "")
            self.aggregator = artifact.get("aggregation_mode", "mean")
            self.embedding_dim = int(artifact.get("embedding_dim", self.embedding_dim))
            self.model_key = str(artifact.get("model_name", self.model_key))

    def encode(self, payload: dict[str, Any]) -> list[float]:
        """Encode a payload into a deterministic embedding vector.

        Args:
            payload: Vision input payload or feature metadata.

        Returns:
            Embedding vector matching the configured model dimension.
        """
        seed = flatten_payload(payload) or "empty_vision_payload"
        return _hash_embedding(f"{self.model_key}:{seed}", self.embedding_dim)

    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return lightweight explainability metadata for a prediction payload."""
        patch_count = int(payload.get("patch_count", 1)) if isinstance(payload, dict) else 1
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "embedding_backend": self.embedding_backend,
            "aggregation_mode": self.aggregator,
            "patch_count": patch_count,
            "source_id": payload.get("sample_id", "") if isinstance(payload, dict) else "",
            "feature_manifest_path": self.feature_manifest_path,
        }

    def predict(self, sample_id: str, payload: dict[str, Any]) -> AgentPrediction:
        """Predict a vision label from the payload using centroid similarity."""
        embedding = self.encode(payload)
        if self.class_centroids:
            raw_scores = {
                label: max(0.0, _cosine_similarity(embedding, centroid))
                for label, centroid in self.class_centroids.items()
            }
            total = sum(raw_scores.values())
            if total > 0:
                scores = {label: round(raw_scores.get(label, 0.0) / total, 4) for label in self.labels}
            else:
                scores = self._scores(payload)
        else:
            scores = self._scores(payload)
        top_label = max(scores, key=scores.get)
        metadata = self.explain(payload)
        metadata.update({"artifact_path": str(self.artifact_path) if self.artifact_path else ""})
        return AgentPrediction(
            modality=self.modality,
            sample_id=sample_id,
            predicted_label=top_label,
            scores=scores,
            embedding=[round(value, 4) for value in embedding[: min(8, len(embedding))]],
            metadata=metadata,
        )
