"""Vision-agent package for foundation-model feature extraction."""

from .aggregator import aggregate_embeddings
from .foundation_models import MODELS, VisionModelSpec, get_embed_dim, get_model_spec, list_model_specs, load_model
from .runtime import VisionAgent

__all__ = [
    "MODELS",
    "VisionAgent",
    "VisionModelSpec",
    "aggregate_embeddings",
    "get_embed_dim",
    "get_model_spec",
    "list_model_specs",
    "load_model",
]
