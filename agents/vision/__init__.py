"""Vision-agent package for foundation-model feature extraction."""

from .aggregator import aggregate_embeddings
from .foundation_models import VisionModelSpec, get_model_spec, list_model_specs
from .runtime import VisionAgent

__all__ = [
    "VisionAgent",
    "VisionModelSpec",
    "aggregate_embeddings",
    "get_model_spec",
    "list_model_specs",
]
