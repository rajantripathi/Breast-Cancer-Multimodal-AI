from __future__ import annotations

from .base_agent import BaseAgent


class LiteratureAgent(BaseAgent):
    modality = "literature"

    def __init__(self, artifact_path: str | None = None) -> None:
        super().__init__(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            ["limited_evidence", "supportive_evidence"],
            artifact_path=artifact_path,
        )
