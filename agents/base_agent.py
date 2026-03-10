from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentPrediction:
    modality: str
    sample_id: str
    predicted_label: str
    scores: dict[str, float]
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    modality = "base"

    def __init__(self, model_name: str, labels: list[str] | None = None, artifact_path: str | None = None) -> None:
        self.model_name = model_name
        self.labels = labels or ["negative", "positive"]
        self.artifact_path = Path(artifact_path) if artifact_path else None
        self.prototypes: dict[str, dict[str, float]] = {}
        if self.artifact_path and self.artifact_path.exists():
            artifact = json.loads(self.artifact_path.read_text())
            self.prototypes = artifact.get("prototypes", {})
            self.labels = artifact.get("labels", self.labels)

    def predict(self, sample_id: str, payload: dict[str, Any]) -> AgentPrediction:
        scores = self._scores(payload)
        top_label = max(scores, key=scores.get)
        return AgentPrediction(
            modality=self.modality,
            sample_id=sample_id,
            predicted_label=top_label,
            scores=scores,
            embedding=[round(scores[label], 4) for label in self.labels[:8]],
            metadata={"model_name": self.model_name, "artifact_path": str(self.artifact_path) if self.artifact_path else ""},
        )

    def _scores(self, payload: dict[str, Any]) -> dict[str, float]:
        text = str(payload)
        if self.prototypes:
            tokens = text.lower().replace("{", " ").replace("}", " ").replace(":", " ").replace(",", " ").split()
            raw_scores = {}
            for label, weights in self.prototypes.items():
                raw_scores[label] = sum(weights.get(token, 0.0) for token in tokens)
            total = sum(raw_scores.values())
            if total > 0:
                return {label: round(raw_scores.get(label, 0.0) / total, 4) for label in self.labels}
        score = min(0.95, max(0.05, (len(text) % 100) / 100))
        if len(self.labels) == 2:
            return {self.labels[0]: round(1 - score, 4), self.labels[-1]: round(score, 4)}
        scores = {label: 0.0 for label in self.labels}
        scores[self.labels[0]] = round(max(0.0, 1 - score), 4)
        scores[self.labels[-1]] = round(score, 4)
        return scores
