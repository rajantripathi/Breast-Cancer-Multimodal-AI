from __future__ import annotations

from .base_agent import BaseAgent


class EHRAgent(BaseAgent):
    modality = "ehr"

    def __init__(self, artifact_path: str | None = None) -> None:
        super().__init__("emilyalsentzer/Bio_ClinicalBERT", ["low_risk", "high_risk"], artifact_path=artifact_path)
