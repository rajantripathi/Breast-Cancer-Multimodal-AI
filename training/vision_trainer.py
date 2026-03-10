from __future__ import annotations

"""Feature-based vision trainer for the Phase 1 registry pipeline."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import load_settings
from data.common import read_json, read_jsonl, write_json

from agents.vision.aggregator import aggregate_embeddings


def _load_embedding(path: str | Path) -> list[float]:
    """Load an embedding vector from a feature artifact."""
    payload = json.loads(Path(path).read_text())
    return [float(value) for value in payload.get("embedding", [])]


def _mean_centroid(vectors: list[list[float]]) -> list[float]:
    """Compute a mean centroid for a set of vectors."""
    return aggregate_embeddings(vectors, pooling="mean")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two equal-width vectors."""
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(width))
    left_norm = sum(left[index] * left[index] for index in range(width)) ** 0.5
    right_norm = sum(right[index] * right[index] for index in range(width)) ** 0.5
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _score_embedding(embedding: list[float], centroids: dict[str, list[float]]) -> dict[str, float]:
    """Score one embedding against label centroids."""
    raw_scores = {label: max(0.0, _cosine_similarity(embedding, centroid)) for label, centroid in centroids.items()}
    total = sum(raw_scores.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(raw_scores))
        return {label: uniform for label in raw_scores}
    return {label: score / total for label, score in raw_scores.items()}


def train_feature_classifier(args: argparse.Namespace) -> Path:
    """Train a centroid-based classifier on extracted vision features."""
    settings = load_settings(args.config)
    output_dir = Path(args.output_dir or settings.output_root / "vision")
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_settings = settings.extras.get("vision", {})
    training_settings = vision_settings.get("training", {})
    selected_model = args.model_key or vision_settings.get("default_model", "uni2")
    default_manifest = settings.output_root / "vision" / "features" / selected_model / "features_manifest.jsonl"
    feature_manifest_path = Path(args.feature_manifest or default_manifest)
    split_path = Path(args.split_path or settings.split_root / "vision_splits.json")
    rows = read_jsonl(feature_manifest_path)
    if args.smoke_test:
        rows = rows[: max(3, min(len(rows), int(vision_settings.get("smoke_limit", 8))))]
    split_manifest = read_json(split_path) if split_path.exists() else {"train": [row["sample_id"] for row in rows], "val": [row["sample_id"] for row in rows]}
    train_ids = set(split_manifest.get("train", []))
    val_ids = set(split_manifest.get("val", []))
    train_rows = [row for row in rows if row["sample_id"] in train_ids] or rows
    val_rows = [row for row in rows if row["sample_id"] in val_ids] or rows[: max(1, len(rows) // 3)]

    label_to_vectors: dict[str, list[list[float]]] = defaultdict(list)
    for row in train_rows:
        label_to_vectors[row["label"]].append(_load_embedding(row["embedding_path"]))
    centroids = {label: _mean_centroid(vectors) for label, vectors in label_to_vectors.items() if vectors}

    predictions = []
    correct = 0
    for row in val_rows:
        embedding = _load_embedding(row["embedding_path"])
        probabilities = _score_embedding(embedding, centroids)
        predicted_label = max(probabilities, key=probabilities.get)
        correct += int(predicted_label == row["label"])
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "true_label": row["label"],
                "predicted_label": predicted_label,
                "probabilities": probabilities,
            }
        )
    accuracy = correct / len(val_rows) if val_rows else 0.0
    artifact = {
        "task": "vision",
        "model_name": selected_model,
        "labels": sorted(centroids),
        "feature_manifest_path": str(feature_manifest_path),
        "split_path": str(split_path),
        "aggregation_mode": training_settings.get("centroid_metric", "cosine"),
        "class_centroids": centroids,
        "metrics": {
            "val_accuracy": round(accuracy, 4),
            "dataset_rows": len(rows),
            "num_train": len(train_rows),
            "num_val": len(val_rows),
            "label_distribution": dict(Counter(row["label"] for row in rows)),
        },
        "predictions": predictions,
    }
    write_json(output_dir / "artifact.json", artifact)
    write_json(output_dir / "summary.json", artifact["metrics"])
    write_json(output_dir / "predictions.json", predictions)
    return output_dir / "artifact.json"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for vision training."""
    parser = argparse.ArgumentParser(description="Train the Phase 1 vision classifier from features")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--feature-manifest", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    """Train the vision centroid classifier and write an artifact."""
    args = build_parser().parse_args()
    path = train_feature_classifier(args)
    print(f"vision artifact written to {path}")


if __name__ == "__main__":
    main()
