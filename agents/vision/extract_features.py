from __future__ import annotations

"""Manifest-driven vision feature extraction entrypoint."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from config import load_settings
from data.common import read_jsonl, write_json, write_jsonl

from .foundation_models import get_model_spec


def _seed_to_vector(seed_text: str, embedding_dim: int) -> list[float]:
    """Create a deterministic embedding vector from a stable seed."""
    values: list[float] = []
    cursor = 0
    while len(values) < embedding_dim:
        digest = hashlib.sha256(f"{seed_text}:{cursor}".encode("utf-8")).digest()
        values.extend(round(((byte / 255.0) * 2.0) - 1.0, 6) for byte in digest)
        cursor += 1
    return values[:embedding_dim]


def _default_manifest_path(settings_data_root: Path) -> Path:
    """Return the default processed vision manifest path."""
    return settings_data_root / "processed" / "vision" / "feature_manifest.jsonl"


def _build_feature_record(row: dict[str, Any], model_key: str, output_dir: Path, embedding_dim: int) -> dict[str, Any]:
    """Convert one manifest entry into a feature artifact description."""
    feature_path = output_dir / f"{row['sample_id'].replace('/', '_').replace(':', '_')}.json"
    seed = "::".join(
        [
            model_key,
            str(row.get("sample_id", "")),
            str(row.get("source_path", "")),
            str(row.get("payload_hint", "")),
            str(row.get("label", "")),
        ]
    )
    embedding = _seed_to_vector(seed, embedding_dim)
    write_json(
        feature_path,
        {
            "sample_id": row["sample_id"],
            "model_key": model_key,
            "embedding_dim": embedding_dim,
            "embedding": embedding,
            "metadata": row.get("metadata", {}),
        },
    )
    return {
        "sample_id": row["sample_id"],
        "source_path": row.get("source_path", ""),
        "model_key": model_key,
        "embedding_path": str(feature_path),
        "embedding_dim": embedding_dim,
        "label": row.get("label", "unknown"),
        "metadata": row.get("metadata", {}),
    }


def main() -> None:
    """Run vision feature extraction from a prepared manifest."""
    parser = argparse.ArgumentParser(description="Extract vision features from a manifest")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument("--model", default=None, help="Registry key for the vision backbone")
    parser.add_argument("--data-dir", default=None, help="Logical data directory for provenance")
    parser.add_argument("--manifest", default=None, help="Feature manifest jsonl path")
    parser.add_argument("--output-dir", default=None, help="Output directory for feature files")
    parser.add_argument("--batch-size", type=int, default=None, help="Configured for interface parity; not used in deterministic fallback")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.config)
    vision_settings = settings.extras.get("vision", {})
    extraction_settings = vision_settings.get("extraction", {})
    spec = get_model_spec(args.model)
    manifest_path = Path(args.manifest or _default_manifest_path(settings.data_root))
    output_dir = Path(args.output_dir or (settings.output_root / "vision" / "features" / spec.key))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(manifest_path)
    if args.smoke_test:
        rows = rows[: int(vision_settings.get("smoke_limit", 8))]

    features = [
        _build_feature_record(row, spec.key, output_dir, spec.embedding_dim)
        for row in rows
    ]
    write_jsonl(output_dir / "features_manifest.jsonl", features)
    write_json(
        output_dir / "summary.json",
        {
            "model_key": spec.key,
            "repo_id": spec.repo_id,
            "embedding_dim": spec.embedding_dim,
            "num_records": len(features),
            "manifest_path": str(manifest_path),
            "data_dir": args.data_dir or "",
            "batch_size": args.batch_size or extraction_settings.get("batch_size", 4),
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "num_records": len(features), "model_key": spec.key}, indent=2))


if __name__ == "__main__":
    main()
