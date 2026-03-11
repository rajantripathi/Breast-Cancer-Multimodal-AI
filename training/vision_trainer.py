from __future__ import annotations

"""Feature-based vision trainer for foundation-model embeddings."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import load_settings
from data.common import read_json, read_jsonl, write_json

from agents.vision.aggregator import aggregate_embeddings
from agents.vision.foundation_models import get_embed_dim


def _load_embedding(path: str | Path) -> list[float]:
    """Load an embedding vector from a feature artifact."""
    import torch

    target = Path(path)
    if target.suffix == ".pt":
        payload = torch.load(target, map_location="cpu")
        embedding = payload.get("embedding", [])
        if hasattr(embedding, "tolist"):
            return [float(value) for value in embedding.tolist()]
        return [float(value) for value in embedding]
    payload = json.loads(target.read_text())
    return [float(value) for value in payload.get("embedding", [])]


def _load_embedding_dim(path: str | Path) -> int:
    """Load the recorded embedding dimension from a feature artifact."""
    import torch

    target = Path(path)
    if target.suffix == ".pt":
        payload = torch.load(target, map_location="cpu")
        if "embedding_dim" in payload:
            return int(payload["embedding_dim"])
        embedding = payload.get("embedding", [])
        return int(embedding.shape[-1] if hasattr(embedding, "shape") else len(embedding))
    payload = json.loads(target.read_text())
    return int(payload.get("embedding_dim", len(payload.get("embedding", []))))


def _mean_centroid(vectors: list[list[float]]) -> list[float]:
    """Compute a mean centroid for a set of vectors."""
    return aggregate_embeddings(vectors, pooling="mean")


def _set_seed(seed: int) -> None:
    """Seed Python and torch RNGs for reproducible training."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_rows(
    args: argparse.Namespace,
    settings: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Path, Path, str, dict[str, Any]]:
    """Resolve feature rows, splits, and trainer config."""
    vision_settings = settings.extras.get("vision", {})
    training_settings = vision_settings.get("training", {})
    selected_model = args.model_key or vision_settings.get("default_model", "uni2")
    default_manifest = settings.output_root / "vision" / "features" / selected_model / "features_manifest.jsonl"
    feature_manifest_path = Path(args.feature_manifest or default_manifest)
    split_path = Path(args.split_path or settings.split_root / "vision_splits.json")
    rows = read_jsonl(feature_manifest_path)
    if args.smoke_test:
        rows = rows[: max(6, min(len(rows), int(vision_settings.get("smoke_limit", 12))))]
    split_manifest = (
        read_json(split_path)
        if split_path.exists()
        else {"train": [row["sample_id"] for row in rows], "val": [row["sample_id"] for row in rows]}
    )
    train_ids = set(split_manifest.get("train", []))
    val_ids = set(split_manifest.get("val", []))
    train_rows = [row for row in rows if row["sample_id"] in train_ids] or rows
    val_rows = [row for row in rows if row["sample_id"] in val_ids] or rows[: max(1, len(rows) // 3)]
    return rows, train_rows, val_rows, feature_manifest_path, split_path, selected_model, training_settings


def _build_label_mapping(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    """Build stable label mappings from dataset rows."""
    labels = sorted({row["label"] for row in rows})
    return labels, {label: index for index, label in enumerate(labels)}


def _build_dataset(
    rows: list[dict[str, Any]],
    label_to_index: dict[str, int],
) -> tuple[Any, Any]:
    """Build tensor datasets from feature rows."""
    import torch

    features = [_load_embedding(row["embedding_path"]) for row in rows]
    labels = [label_to_index[row["label"]] for row in rows]
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return feature_tensor, label_tensor


def _class_weights(train_rows: list[dict[str, Any]], labels: list[str]) -> Any:
    """Compute inverse-frequency class weights."""
    import torch

    counts = Counter(row["label"] for row in train_rows)
    weights = []
    for label in labels:
        count = max(1, counts.get(label, 1))
        weights.append(1.0 / count)
    total = sum(weights) or 1.0
    scaled = [weight * len(weights) / total for weight in weights]
    return torch.tensor(scaled, dtype=torch.float32)


def _build_model(input_dim: int, num_classes: int) -> Any:
    """Build the classifier head for vision embeddings."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


def _per_class_metrics(
    true_indices: list[int],
    pred_indices: list[int],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-class accuracy, precision, and recall."""
    metrics: dict[str, dict[str, float]] = {}
    for class_index, label in enumerate(labels):
        tp = sum(1 for true, pred in zip(true_indices, pred_indices) if true == class_index and pred == class_index)
        fp = sum(1 for true, pred in zip(true_indices, pred_indices) if true != class_index and pred == class_index)
        fn = sum(1 for true, pred in zip(true_indices, pred_indices) if true == class_index and pred != class_index)
        total = sum(1 for true in true_indices if true == class_index)
        accuracy = tp / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics[label] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    return metrics


def _run_training(
    train_features: Any,
    train_labels: Any,
    val_features: Any,
    val_labels: Any,
    class_weights: Any,
    learning_rate: float,
    patience: int,
    epochs: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Train one classifier candidate and return the best checkpoint summary."""
    import torch
    import torch.nn.functional as functional

    _set_seed(seed)
    model = _build_model(train_features.shape[-1], int(train_labels.max().item()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_state = None
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_features)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_features)
            val_loss = criterion(val_logits, val_labels).item()
            val_predictions = val_logits.argmax(dim=-1)
            accuracy = (val_predictions == val_labels).float().mean().item()

        improved = accuracy > best_accuracy or (abs(accuracy - best_accuracy) < 1e-8 and val_loss < best_loss)
        if improved:
            best_accuracy = accuracy
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    assert best_state is not None, "vision trainer failed to capture a checkpoint"
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(val_features)
        final_probabilities = functional.softmax(final_logits, dim=-1).cpu()
        final_predictions = final_logits.argmax(dim=-1).cpu()

    return {
        "state_dict": best_state,
        "best_accuracy": best_accuracy,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "probabilities": final_probabilities,
        "predictions": final_predictions,
    }


def train_feature_classifier(args: argparse.Namespace) -> Path:
    """Train a neural classifier on extracted vision features."""
    import torch

    settings = load_settings(args.config)
    output_dir = Path(args.output_dir or settings.output_root / "vision")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, train_rows, val_rows, feature_manifest_path, split_path, selected_model, training_settings = _resolve_rows(args, settings)
    expected_dim = get_embed_dim(selected_model)
    observed_dim = _load_embedding_dim(train_rows[0]["embedding_path"]) if train_rows else expected_dim
    assert observed_dim == expected_dim, f"vision embedding dim mismatch: expected {expected_dim}, got {observed_dim}"

    feature_dir = feature_manifest_path.parent
    print(f"vision trainer feature directory: {feature_dir}")
    print(f"vision trainer embedding dimension: {observed_dim}")
    if selected_model == "uni2":
        assert "outputs/vision/features/uni2" in str(feature_dir), (
            f"vision trainer expected UNI2 features but got {feature_dir}"
        )

    labels, label_to_index = _build_label_mapping(rows)
    train_features, train_labels = _build_dataset(train_rows, label_to_index)
    val_features, val_labels = _build_dataset(val_rows, label_to_index)
    class_weights = _class_weights(train_rows, labels)

    lr_candidates = [float(value) for value in training_settings.get("lr_sweep", [1e-3, 1e-4, 1e-5])]
    if args.smoke_test:
        lr_candidates = [1e-4]
    patience = int(training_settings.get("patience", 20))
    epochs = int(training_settings.get("epochs", 100 if not args.smoke_test else 8))
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    best_run: dict[str, Any] | None = None
    for learning_rate in lr_candidates:
        run = _run_training(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            class_weights=class_weights,
            learning_rate=learning_rate,
            patience=patience,
            epochs=epochs,
            device=device,
            seed=args.seed,
        )
        run["learning_rate"] = learning_rate
        if best_run is None or run["best_accuracy"] > best_run["best_accuracy"] or (
            abs(run["best_accuracy"] - best_run["best_accuracy"]) < 1e-8 and run["best_loss"] < best_run["best_loss"]
        ):
            best_run = run

    assert best_run is not None, "vision trainer failed to produce a run"

    class_centroids: dict[str, list[list[float]]] = defaultdict(list)
    for row in train_rows:
        class_centroids[row["label"]].append(_load_embedding(row["embedding_path"]))
    centroid_payload = {
        label: _mean_centroid(vectors) for label, vectors in class_centroids.items() if vectors
    }

    probabilities = best_run["probabilities"].tolist()
    pred_indices = best_run["predictions"].tolist()
    true_indices = val_labels.tolist()
    predictions = []
    for row, true_index, predicted_index, probability_row in zip(val_rows, true_indices, pred_indices, probabilities):
        probability_map = {
            label: round(float(probability_row[label_to_index[label]]), 4) for label in labels
        }
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "true_label": labels[true_index],
                "predicted_label": labels[predicted_index],
                "probabilities": probability_map,
            }
        )

    per_class = _per_class_metrics(true_indices, pred_indices, labels)
    accuracy = sum(int(true == pred) for true, pred in zip(true_indices, pred_indices)) / len(true_indices) if true_indices else 0.0
    artifact = {
        "task": "vision",
        "model_name": selected_model,
        "embedding_dim": expected_dim,
        "labels": labels,
        "feature_manifest_path": str(feature_manifest_path),
        "feature_dir": str(feature_dir),
        "split_path": str(split_path),
        "aggregation_mode": training_settings.get("centroid_metric", "cosine"),
        "class_centroids": centroid_payload,
        "classifier_state": {key: value.tolist() for key, value in best_run["state_dict"].items()},
        "classifier_architecture": {
            "input_dim": expected_dim,
            "hidden_dim": 512,
            "dropout": 0.3,
            "num_classes": len(labels),
        },
        "metrics": {
            "val_accuracy": round(accuracy, 4),
            "dataset_rows": len(rows),
            "num_train": len(train_rows),
            "num_val": len(val_rows),
            "label_distribution": dict(Counter(row["label"] for row in rows)),
            "selected_learning_rate": best_run["learning_rate"],
            "lr_sweep": lr_candidates,
            "best_epoch": best_run["best_epoch"],
            "patience": patience,
            "per_class_metrics": per_class,
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
    """Train the vision classifier and write an artifact."""
    args = build_parser().parse_args()
    path = train_feature_classifier(args)
    print(f"vision artifact written to {path}")


if __name__ == "__main__":
    main()
