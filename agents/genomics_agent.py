from __future__ import annotations

from .base_agent import BaseAgent


class GenomicsAgent(BaseAgent):
    modality = "genomics"

    def __init__(self, artifact_path: str | None = None) -> None:
        super().__init__("zhihan1996/DNA_bert_6", ["benign_variant", "pathogenic_variant"], artifact_path=artifact_path)
