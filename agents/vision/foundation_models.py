from __future__ import annotations

"""Registry for supported vision foundation models."""

from dataclasses import dataclass
from pathlib import Path

from config import load_settings

DEFAULT_VISION_MODELS = {
    "uni2": {
        "repo_id": "MahmoodLab/UNI2-h",
        "embedding_dim": 1536,
        "gated": True,
        "input_kind": "slide_tiles",
        "cache_subdir": "vision/uni2",
    },
    "conch": {
        "repo_id": "MahmoodLab/CONCH",
        "embedding_dim": 512,
        "gated": True,
        "input_kind": "slide_tiles",
        "cache_subdir": "vision/conch",
    },
    "virchow2": {
        "repo_id": "paige-ai/Virchow2",
        "embedding_dim": 1280,
        "gated": True,
        "input_kind": "slide_tiles",
        "cache_subdir": "vision/virchow2",
    },
    "benchmark_stub": {
        "repo_id": "local/benchmark-stub",
        "embedding_dim": 256,
        "gated": False,
        "input_kind": "image",
        "cache_subdir": "vision/benchmark_stub",
    },
}


@dataclass(frozen=True)
class VisionModelSpec:
    """Metadata describing a supported vision backbone."""

    key: str
    repo_id: str
    embedding_dim: int
    gated: bool
    input_kind: str
    cache_subdir: str

    def cache_dir(self) -> Path:
        """Return the resolved cache directory for this model."""
        settings = load_settings()
        return settings.model_cache_dir / self.cache_subdir


def list_model_specs() -> list[VisionModelSpec]:
    """Return all registered vision backbones."""
    settings = load_settings()
    models = settings.extras.get("vision", {}).get("models", {}) or DEFAULT_VISION_MODELS
    return [
        VisionModelSpec(
            key=key,
            repo_id=str(config.get("repo_id", "")),
            embedding_dim=int(config.get("embedding_dim", 256)),
            gated=bool(config.get("gated", False)),
            input_kind=str(config.get("input_kind", "image")),
            cache_subdir=str(config.get("cache_subdir", f"vision/{key}")),
        )
        for key, config in models.items()
    ]


def get_model_spec(model_key: str | None = None) -> VisionModelSpec:
    """Resolve a model spec by key, falling back to the configured default."""
    settings = load_settings()
    vision_settings = settings.extras.get("vision", {})
    selected_key = model_key or str(vision_settings.get("default_model", "uni2"))
    specs = {spec.key: spec for spec in list_model_specs()}
    if selected_key not in specs:
        raise KeyError(f"unknown vision model: {selected_key}")
    return specs[selected_key]
