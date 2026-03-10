from __future__ import annotations

"""Registry and loader utilities for pathology vision foundation models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

from config import load_settings


@dataclass(frozen=True)
class VisionModelSpec:
    """Metadata describing a supported vision foundation model."""

    name: str
    hub: str
    embed_dim: int
    architecture: str
    gated: bool
    access_url: str

    def cache_dir(self) -> Path:
        """Return the local cache directory for the model.

        Returns:
            Cache directory under the configured model cache root.
        """
        settings = load_settings()
        return settings.model_cache_dir / "vision" / self.name


MODELS: dict[str, dict[str, Any]] = {
    "uni2": {
        "hub": "MahmoodLab/UNI2-h",
        "embed_dim": 1536,
        "architecture": "ViT-H",
        "gated": True,
        "access_url": "https://huggingface.co/MahmoodLab/UNI2-h",
    },
    "conch": {
        "hub": "MahmoodLab/CONCH",
        "embed_dim": 512,
        "architecture": "vision-language",
        "gated": True,
        "access_url": "https://huggingface.co/MahmoodLab/CONCH",
    },
    "virchow2": {
        "hub": "paige-ai/Virchow2",
        "embed_dim": 1280,
        "architecture": "ViT-H",
        "gated": True,
        "access_url": "https://huggingface.co/paige-ai/Virchow2",
    },
    "ctranspath": {
        "hub": "xiyuez/ctranspath",
        "embed_dim": 768,
        "architecture": "Swin Transformer",
        "gated": False,
        "access_url": "https://huggingface.co/xiyuez/ctranspath",
    },
}


def get_model_spec(name: str | None = None) -> VisionModelSpec:
    """Resolve a model specification by name.

    Args:
        name: Requested registry key. If omitted, uses the configured default.

    Returns:
        A structured model specification.

    Raises:
        KeyError: If the model name is unknown.
    """
    settings = load_settings()
    requested_name = name or str(settings.extras.get("vision", {}).get("default_model", "uni2"))
    if requested_name not in MODELS:
        raise KeyError(f"unknown vision model: {requested_name}")
    config = MODELS[requested_name]
    return VisionModelSpec(
        name=requested_name,
        hub=str(config["hub"]),
        embed_dim=int(config["embed_dim"]),
        architecture=str(config["architecture"]),
        gated=bool(config["gated"]),
        access_url=str(config["access_url"]),
    )


def list_model_specs() -> list[VisionModelSpec]:
    """Return all registered model specifications.

    Returns:
        Registry entries in declaration order.
    """
    return [get_model_spec(name) for name in MODELS]


def get_embed_dim(name: str) -> int:
    """Return the embedding dimension for a registered model.

    Args:
        name: Registry key for the vision model.

    Returns:
        The output embedding dimension.
    """
    return get_model_spec(name).embed_dim


def _build_transform(model: Any) -> Any:
    """Build an inference transform for a timm model.

    Args:
        model: Loaded timm-compatible model.

    Returns:
        Callable image transform.
    """
    from timm.data import create_transform, resolve_data_config

    data_config = resolve_data_config(model.pretrained_cfg if hasattr(model, "pretrained_cfg") else {}, model=model)
    return create_transform(**data_config, is_training=False)


def _load_from_timm(spec: VisionModelSpec) -> Any:
    """Load a pretrained encoder from timm or timm HF hub integration.

    Args:
        spec: Selected model specification.

    Returns:
        Loaded model instance.
    """
    import timm

    kwargs = {"pretrained": True, "num_classes": 0}
    target = f"hf-hub:{spec.hub}"
    try:
        return timm.create_model(target, **kwargs)
    except Exception:
        # Some timm versions or hubs require explicit local cache use.
        return timm.create_model(target, pretrained=True)


def _emit_access_error(spec: VisionModelSpec, exc: Exception) -> None:
    """Print a clear gated-access error for a failed model load.

    Args:
        spec: Model specification that failed.
        exc: Underlying exception.
    """
    print(
        (
            f"Failed to load gated model '{spec.name}' from {spec.hub}. "
            f"Request access at {spec.access_url}. Original error: {exc}"
        ),
        file=sys.stderr,
    )


def _freeze_encoder(model: Any) -> Any:
    """Freeze a model for inference-only feature extraction.

    Args:
        model: Loaded model instance.

    Returns:
        The same model in eval mode with gradients disabled.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_model(name: str) -> tuple[Any, Any]:
    """Load a frozen pathology foundation model and its transform.

    Args:
        name: Registry key for the requested model.

    Returns:
        Tuple of `(model, transform)`.

    Raises:
        RuntimeError: If `timm` is unavailable or all eligible load attempts fail.
    """
    spec = get_model_spec(name)
    try:
        model = _load_from_timm(spec)
        return _freeze_encoder(model), _build_transform(model)
    except ImportError as exc:
        raise RuntimeError("timm is required to load pathology foundation models") from exc
    except Exception as exc:
        if spec.gated:
            _emit_access_error(spec, exc)
            if spec.name != "ctranspath":
                fallback = get_model_spec("ctranspath")
                print(
                    f"Falling back to open model '{fallback.name}' from {fallback.access_url}",
                    file=sys.stderr,
                )
                model = _load_from_timm(fallback)
                return _freeze_encoder(model), _build_transform(model)
        raise RuntimeError(f"failed to load model '{spec.name}' from {spec.hub}: {exc}") from exc
