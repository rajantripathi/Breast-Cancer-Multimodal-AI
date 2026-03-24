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
    timm_model_name: str | None = None
    timm_kwargs: dict[str, Any] | None = None

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
        "timm_kwargs": {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 5.33334,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": "SwiGLUPacked",
            "act_layer": "SiLU",
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
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
    "virchow": {
        "hub": "paige-ai/Virchow",
        "embed_dim": 1280,
        "architecture": "ViT-H",
        "gated": True,
        "access_url": "https://huggingface.co/paige-ai/Virchow",
        "timm_model_name": "hf-hub:paige-ai/Virchow",
    },
    "ctranspath": {
        "hub": "1aurent/swin_tiny_patch4_window7_224.CTransPath",
        "embed_dim": 768,
        "architecture": "Swin Transformer with ConvStem",
        "gated": False,
        "access_url": "https://huggingface.co/1aurent/swin_tiny_patch4_window7_224.CTransPath",
        "timm_model_name": "hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
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
        timm_model_name=str(config.get("timm_model_name")) if config.get("timm_model_name") else None,
        timm_kwargs=dict(config.get("timm_kwargs", {})) if config.get("timm_kwargs") else None,
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
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "timm is required for foundation model extraction. "
            "Install: pip install timm>=1.0.3 huggingface-hub>=0.23.0"
        ) from exc

    kwargs = _resolve_timm_kwargs(spec)
    target = spec.timm_model_name or f"hf-hub:{spec.hub}"
    if spec.name == "ctranspath":
        kwargs["embed_layer"] = _build_ctranspath_conv_stem()
    try:
        return timm.create_model(target, **kwargs)
    except Exception:
        # Some timm versions or hubs require explicit local cache use.
        retry_kwargs = dict(kwargs)
        if spec.name == "ctranspath":
            retry_kwargs["embed_layer"] = _build_ctranspath_conv_stem()
        return timm.create_model(target, **retry_kwargs)


def _load_conch(spec: VisionModelSpec) -> tuple[Any, Any]:
    """Load CONCH with its custom open_clip loader.

    Args:
        spec: Selected model specification.

    Returns:
        Tuple of `(model, transform)`.
    """
    import os

    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        raise RuntimeError(
            "CONCH extraction requires the MahmoodLab CONCH package. "
            "Install: pip install git+https://github.com/Mahmoodlab/CONCH.git"
        ) from exc

    token = os.environ.get("CONCH_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("CONCH extraction requires CONCH_HF_TOKEN or HF_TOKEN to be set")
    hub_ref = spec.hub if spec.hub != "MahmoodLab/CONCH" else "MahmoodLab/conch"
    model, transform = create_model_from_pretrained(
        "conch_ViT-B-16",
        f"hf_hub:{hub_ref}",
        hf_auth_token=token,
    )
    return model, transform


def _resolve_timm_kwargs(spec: VisionModelSpec) -> dict[str, Any]:
    """Return timm creation kwargs for a registered model.

    Args:
        spec: Selected model specification.

    Returns:
        Keyword arguments for `timm.create_model`.
    """
    kwargs: dict[str, Any] = {"pretrained": True, "num_classes": 0}
    if not spec.timm_kwargs:
        return kwargs

    try:
        import timm
        import torch
    except ImportError:
        return {**kwargs, **spec.timm_kwargs}

    resolved = dict(spec.timm_kwargs)
    mlp_layer = resolved.get("mlp_layer")
    if mlp_layer == "SwiGLUPacked":
        resolved["mlp_layer"] = timm.layers.SwiGLUPacked
    act_layer = resolved.get("act_layer")
    if act_layer == "SiLU":
        resolved["act_layer"] = torch.nn.SiLU
    return {**kwargs, **resolved}


def _build_ctranspath_conv_stem() -> type[Any]:
    """Return the custom ConvStem layer required by timm CTransPath checkpoints.

    Returns:
        A `torch.nn.Module` class implementing the patch embedding stem.
    """
    import torch.nn as nn
    from timm.layers.helpers import to_2tuple

    class ConvStem(nn.Module):
        """Patch embedding stem used by CTransPath."""

        def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Any | None = None,
            **_: Any,
        ) -> None:
            super().__init__()
            assert patch_size == 4, "Patch size must be 4 for CTransPath"
            assert embed_dim % 8 == 0, "Embedding dimension must be divisible by 8"

            image_size = to_2tuple(img_size)
            patch = to_2tuple(patch_size)
            self.img_size = image_size
            self.patch_size = patch
            self.grid_size = (image_size[0] // patch[0], image_size[1] // patch[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]

            stem_layers: list[nn.Module] = []
            input_dim = in_chans
            output_dim = embed_dim // 8
            for _index in range(2):
                stem_layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
                stem_layers.append(nn.BatchNorm2d(output_dim))
                stem_layers.append(nn.ReLU(inplace=True))
                input_dim = output_dim
                output_dim *= 2
            stem_layers.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
            self.proj = nn.Sequential(*stem_layers)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        def forward(self, x: Any) -> Any:
            """Project an image tensor into patch embeddings."""
            _, _, height, width = x.shape
            assert (height, width) == self.img_size, (
                f"Input image size ({height}x{width}) does not match model "
                f"({self.img_size[0]}x{self.img_size[1]})."
            )
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            return self.norm(x)

    return ConvStem


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
        RuntimeError: If required dependencies are unavailable or the model cannot be loaded.
    """
    spec = get_model_spec(name)
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for foundation model extraction. "
            "Install: pip install timm>=1.0.3 huggingface-hub>=0.23.0"
        ) from exc
    try:
        if spec.name == "conch":
            model, transform = _load_conch(spec)
            return _freeze_encoder(model), transform
        model = _load_from_timm(spec)
        return _freeze_encoder(model), _build_transform(model)
    except Exception as exc:
        if spec.gated:
            _emit_access_error(spec, exc)
        raise RuntimeError(f"failed to load model '{spec.name}' from {spec.hub}: {exc}") from exc
