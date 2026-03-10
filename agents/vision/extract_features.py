from __future__ import annotations

"""Feature extraction entrypoint for pathology vision foundation models."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from config import load_settings
from data.common import read_jsonl, write_json, write_jsonl

from .foundation_models import get_embed_dim, get_model_spec, load_model


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _default_manifest_path(settings_data_root: Path) -> Path:
    """Return the default processed vision feature manifest path.

    Args:
        settings_data_root: Root data directory from settings.

    Returns:
        Default manifest path for vision feature extraction.
    """
    return settings_data_root / "processed" / "vision" / "feature_manifest.jsonl"


def _is_image(path: Path) -> bool:
    """Check whether a path looks like a supported image file.

    Args:
        path: Candidate filesystem path.

    Returns:
        True if the suffix is a supported image type.
    """
    return path.suffix.lower() in IMAGE_SUFFIXES


def _resolve_image_path(row: dict[str, Any], data_dir: Path) -> Path | None:
    """Resolve an image path from a manifest row.

    Args:
        row: Manifest row with a `source_path`.
        data_dir: User-provided data directory.

    Returns:
        Absolute image path if one can be resolved, otherwise `None`.
    """
    source_path = str(row.get("source_path", "")).split("::", maxsplit=1)[0].strip()
    if not source_path:
        return None
    candidate = Path(source_path)
    if candidate.exists() and _is_image(candidate):
        return candidate
    joined = (data_dir / source_path).resolve()
    if joined.exists() and _is_image(joined):
        return joined
    return None


def _scan_images(data_dir: Path) -> list[dict[str, Any]]:
    """Scan a directory recursively for image files.

    Args:
        data_dir: Root directory containing tiles or images.

    Returns:
        Synthetic manifest rows discovered from the filesystem.
    """
    rows = []
    for image_path in sorted(data_dir.rglob("*")):
        if image_path.is_file() and _is_image(image_path):
            rows.append(
                {
                    "sample_id": image_path.stem,
                    "source_path": str(image_path),
                    "label": "unknown",
                    "metadata": {"source_path": str(image_path), "scan_origin": str(data_dir)},
                }
            )
    return rows


def _fallback_embedding(seed_text: str, embedding_dim: int) -> list[float]:
    """Create a deterministic fallback embedding when an image cannot be loaded.

    Args:
        seed_text: Stable seed text for deterministic output.
        embedding_dim: Requested embedding dimension.

    Returns:
        Dense fallback embedding values.
    """
    values: list[float] = []
    cursor = 0
    while len(values) < embedding_dim:
        digest = hashlib.sha256(f"{seed_text}:{cursor}".encode("utf-8")).digest()
        values.extend(((byte / 255.0) * 2.0) - 1.0 for byte in digest)
        cursor += 1
    return [round(value, 6) for value in values[:embedding_dim]]


def _extract_tensor_embedding(model: Any, image_path: Path, transform: Any) -> list[float]:
    """Run a forward pass for one image and return its embedding.

    Args:
        model: Loaded vision encoder.
        image_path: Path to the input image.
        transform: Inference transform callable.

    Returns:
        Flattened embedding vector.
    """
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch = batch.to(device)
    with torch.no_grad():
        outputs = model(batch)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    return outputs.detach().cpu().reshape(-1).tolist()


def _write_embedding_file(
    row: dict[str, Any],
    embedding: list[float],
    model_key: str,
    output_dir: Path,
    backend: str,
) -> dict[str, Any]:
    """Persist one extracted embedding and return its manifest row.

    Args:
        row: Input manifest row.
        embedding: Extracted embedding vector.
        model_key: Registry key for the model.
        output_dir: Feature output directory.
        backend: Extraction backend description.

    Returns:
        Output manifest row for the embedding artifact.
    """
    import torch

    file_stem = row["sample_id"].replace("/", "_").replace(":", "_")
    feature_path = output_dir / f"{file_stem}.pt"
    tensor = torch.tensor(embedding, dtype=torch.float32)
    torch.save(
        {
            "sample_id": row["sample_id"],
            "model_key": model_key,
            "embedding_dim": tensor.shape[-1],
            "embedding": tensor,
            "metadata": row.get("metadata", {}),
            "backend": backend,
        },
        feature_path,
    )
    return {
        "sample_id": row["sample_id"],
        "source_path": row.get("source_path", ""),
        "model_key": model_key,
        "embedding_path": str(feature_path),
        "embedding_dim": int(tensor.shape[-1]),
        "label": row.get("label", "unknown"),
        "metadata": {**row.get("metadata", {}), "backend": backend},
    }


def main() -> None:
    """Extract embeddings for all images under a data directory or manifest."""
    parser = argparse.ArgumentParser(description="Extract vision features from image tiles or manifests")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument("--model", default="uni2", help="Registry key for the vision backbone")
    parser.add_argument("--data-dir", required=True, help="Directory containing images or tiles")
    parser.add_argument("--manifest", default=None, help="Optional manifest jsonl path")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt feature files")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size placeholder for interface compatibility")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers to advertise in logs")
    parser.add_argument(
        "--allow-deterministic-fallback",
        action="store_true",
        help="Use deterministic placeholder embeddings only for development or debugging",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else _default_manifest_path(settings.data_root)

    spec = get_model_spec(args.model)
    embed_dim = get_embed_dim(spec.name)
    print(
        json.dumps(
            {
                "event": "vision_feature_extraction_start",
                "model_name": spec.name,
                "hub": spec.hub,
                "embed_dim": embed_dim,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
            indent=2,
        )
    )

    rows = read_jsonl(manifest_path) if manifest_path.exists() else _scan_images(data_dir)
    if args.smoke_test:
        rows = rows[:8]

    model = None
    transform = None
    model_backend = "foundation_model"
    try:
        model, transform = load_model(spec.name)
    except Exception as exc:
        if not args.allow_deterministic_fallback:
            raise RuntimeError(
                "Foundation model extraction failed. "
                "Install: pip install timm>=1.0.3 huggingface-hub>=0.23.0\n"
                "To use deterministic fallback (NOT for production), pass "
                "--allow-deterministic-fallback"
            ) from exc
        print(
            "WARNING: timm or model loading failed. Using deterministic fallback. "
            "These are NOT real foundation model embeddings. "
            f"Original error for {spec.name}: {exc}"
        )
        model_backend = "deterministic_fallback"

    feature_rows = []
    for index, row in enumerate(rows, start=1):
        image_path = _resolve_image_path(row, data_dir)
        if model is not None and transform is not None and image_path is not None:
            embedding = _extract_tensor_embedding(model, image_path, transform)
            backend = model_backend
        else:
            if not args.allow_deterministic_fallback:
                if image_path is None:
                    raise RuntimeError(
                        f"Could not resolve a readable image path for sample '{row.get('sample_id', 'unknown')}'. "
                        "Pass --allow-deterministic-fallback only for development debugging."
                    )
                raise RuntimeError(
                    f"Foundation model extraction unavailable for sample '{row.get('sample_id', 'unknown')}'. "
                    "Pass --allow-deterministic-fallback only for development debugging."
                )
            seed = "::".join(
                [
                    spec.name,
                    str(row.get("sample_id", "")),
                    str(row.get("source_path", "")),
                    str(row.get("label", "")),
                ]
            )
            embedding = _fallback_embedding(seed, embed_dim)
            backend = "deterministic_fallback"
        feature_rows.append(_write_embedding_file(row, embedding, spec.name, output_dir, backend))
        if index % 100 == 0:
            print(f"processed {index} images for model={spec.name}")

    write_jsonl(output_dir / "features_manifest.jsonl", feature_rows)
    write_json(
        output_dir / "summary.json",
        {
            "model_key": spec.name,
            "repo_id": spec.hub,
            "embedding_dim": embed_dim,
            "num_records": len(feature_rows),
            "manifest_path": str(manifest_path),
            "data_dir": str(data_dir),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "num_records": len(feature_rows), "model_key": spec.name}, indent=2))


if __name__ == "__main__":
    main()
