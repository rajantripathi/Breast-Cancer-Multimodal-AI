from __future__ import annotations

from .base_agent import BaseAgent


class VisionAgent(BaseAgent):
    modality = "vision"

    def __init__(self, artifact_path: str | None = None) -> None:
        super().__init__(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            ["normal", "benign", "malignant"],
            artifact_path=artifact_path,
        )
