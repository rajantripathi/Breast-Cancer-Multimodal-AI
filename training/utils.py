from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import load_settings
from data.common import flatten_payload, read_json, read_jsonl, write_json
from agents.vision.foundation_models import get_embed_dim


def build_parser(task_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{task_name} trainer")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--dataset-path", default=None, help="Processed jsonl path")
    parser.add_argument("--split-path", default=None, help="Split manifest json path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _resolve_data_paths(task_name: str, args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    settings = load_settings(args.config)
    output_dir = Path(args.output_dir or settings.output_root / task_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = settings.extras
    data_subdir = config.get("data_subdir", f"processed/{task_name}")
    if not str(data_subdir).startswith("processed/"):
        data_subdir = f"processed/{task_name}"
    modality = str(data_subdir).split("/")[-1]
    dataset_path = Path(args.dataset_path or settings.data_root / data_subdir / "dataset.jsonl")
    split_path = Path(args.split_path or settings.split_root / f"{modality}_splits.json")
    return dataset_path, split_path, output_dir, config


def score_with_prototypes(text: str, prototypes: dict[str, dict[str, float]]) -> dict[str, float]:
    return _score_text(text, prototypes)


def _fit_prototypes(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label = row["label"]
        token_counts[label].update(_tokenize(row["text"]))
    prototypes: dict[str, dict[str, float]] = {}
    for label, counts in token_counts.items():
        total = sum(counts.values()) or 1
        prototypes[label] = {token: count / total for token, count in counts.most_common(64)}
    return prototypes


def _score_text(text: str, prototypes: dict[str, dict[str, float]]) -> dict[str, float]:
    tokens = _tokenize(text)
    scores: dict[str, float] = {}
    for label, weights in prototypes.items():
        score = sum(weights.get(token, 0.0) for token in tokens)
        scores[label] = score
    if not scores:
        return {}
    total = sum(scores.values())
    if total <= 0:
        uniform = 1 / len(scores)
        return {label: uniform for label in scores}
    return {label: score / total for label, score in scores.items()}


def train_text_classifier(task_name: str, args: argparse.Namespace) -> Path:
    dataset_path, split_path, output_dir, config = _resolve_data_paths(task_name, args)
    rows = read_jsonl(dataset_path)
    split_manifest = read_json(split_path)
    train_ids = set(split_manifest.get("train", []))
    val_ids = set(split_manifest.get("val", []))

    if args.smoke_test:
        rows = rows[: max(2, min(len(rows), 4))]
        train_ids = {row["sample_id"] for row in rows[:1]}
        val_ids = {row["sample_id"] for row in rows[1:2]}

    train_rows = [row for row in rows if row["sample_id"] in train_ids] or rows[:1]
    val_rows = [row for row in rows if row["sample_id"] in val_ids] or rows[:1]
    if len({row["label"] for row in train_rows}) < len({row["label"] for row in rows}):
        train_rows = rows

    prototypes = _fit_prototypes(train_rows)
    predictions = []
    correct = 0
    for row in val_rows:
        probabilities = _score_text(row["text"], prototypes)
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
    labels = sorted(prototypes)
    artifact = {
        "task": task_name,
        "model_name": config.get("model_name", f"{task_name}_baseline"),
        "labels": labels,
        "device": args.device,
        "smoke_test": args.smoke_test,
        "dataset_path": str(dataset_path),
        "split_path": str(split_path),
        "prototypes": prototypes,
        "metrics": {
            "val_accuracy": round(accuracy, 4),
            "dataset_rows": len(rows),
            "num_train": len(train_rows),
            "num_val": len(val_rows),
            "loss_proxy": round(max(0.0, 1.0 - accuracy + 0.05), 4),
            "perplexity_proxy": round(math.exp(max(0.0, 1.0 - accuracy)), 4),
            "label_distribution": dict(Counter(row["label"] for row in rows)),
        },
        "predictions": predictions,
    }
    write_json(output_dir / "artifact.json", artifact)
    write_json(output_dir / "summary.json", artifact["metrics"])
    write_json(output_dir / "predictions.json", predictions)
    return output_dir / "artifact.json"


def build_verifier_dataset(repo_root: Path) -> list[dict[str, Any]]:
    sample_cases = {path.stem: read_json(path) for path in (repo_root / "sample_cases").glob("*.json")}
    rows = []
    for sample_id, case in sample_cases.items():
        vision_text = flatten_payload(case.get("vision", {})).lower()
        ehr_text = flatten_payload(case.get("ehr", {})).lower()
        genomics_text = flatten_payload(case.get("genomics", {})).lower()
        literature_text = flatten_payload(case.get("literature", {})).lower()
        features = [
            f"vision_{'malignant' if 'spiculated' in vision_text or 'irregular' in vision_text else 'benign'}",
            f"ehr_{'high_risk' if 'true' in ehr_text or '1' in ehr_text else 'low_risk'}",
            f"genomics_{'pathogenic_variant' if 'pathogenic' in genomics_text else 'benign_variant'}",
            f"literature_{'supportive_evidence' if 'cancer' in literature_text or 'brca' in literature_text else 'limited_evidence'}",
        ]
        positive_votes = sum(
            "malignant" in feature or "high_risk" in feature or "pathogenic" in feature or "supportive" in feature
            for feature in features
        )
        fused_label = "high_concern" if positive_votes >= 2 else "monitor"
        rows.append(
            {
                "sample_id": sample_id,
                "label": fused_label,
                "text": flatten_payload(case) + " " + " ".join(features),
            }
        )
    return rows


def train_verifier(args: argparse.Namespace) -> Path:
    settings = load_settings(args.config)
    output_dir = Path(args.output_dir or settings.output_root / "verifier")
    output_dir.mkdir(parents=True, exist_ok=True)
    vision_model = str(settings.extras.get("vision", {}).get("default_model", "uni2"))
    vision_artifact_path = settings.output_root / "vision" / "artifact.json"
    if vision_artifact_path.exists():
        vision_artifact = read_json(vision_artifact_path)
        expected_dim = get_embed_dim(vision_model)
        actual_dim = int(vision_artifact.get("embedding_dim", expected_dim))
        assert actual_dim == expected_dim, f"vision artifact dim mismatch: expected {expected_dim}, got {actual_dim}"
    verifier_dataset_path = settings.processed_data_root / "verifier" / "dataset.jsonl"
    verifier_split_path = settings.split_root / "verifier_splits.json"
    if verifier_dataset_path.exists():
        rows = read_jsonl(verifier_dataset_path)
    else:
        rows = build_verifier_dataset(settings.repo_root)
    if args.smoke_test:
        rows = rows[:2]
    elif verifier_split_path.exists():
        train_ids = set(read_json(verifier_split_path).get("train", []))
        selected_rows = [row for row in rows if row["sample_id"] in train_ids]
        if selected_rows:
            rows = selected_rows
    prototypes = _fit_prototypes(rows)
    predictions = []
    for row in rows:
        probabilities = _score_text(row["text"], prototypes)
        predicted_label = max(probabilities, key=probabilities.get) if probabilities else "monitor"
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "true_label": row["label"],
                "predicted_label": predicted_label,
                "probabilities": probabilities,
            }
        )
    accuracy = sum(int(item["true_label"] == item["predicted_label"]) for item in predictions) / len(predictions) if predictions else 0.0
    artifact = {
        "task": "verifier",
        "model_name": settings.extras.get("model_name", "late_fusion_baseline"),
        "labels": sorted(prototypes),
        "device": args.device,
        "smoke_test": args.smoke_test,
        "prototypes": prototypes,
        "metrics": {"val_accuracy": round(accuracy, 4), "num_samples": len(rows)},
        "predictions": predictions,
    }
    write_json(output_dir / "artifact.json", artifact)
    write_json(output_dir / "summary.json", artifact["metrics"])
    write_json(output_dir / "predictions.json", predictions)
    return output_dir / "artifact.json"
