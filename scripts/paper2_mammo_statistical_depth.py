from __future__ import annotations

"""Inference-only Stage 1 statistical-depth analysis for mammography."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.mammography.training.train_screener_legacy import (
    MammographyExamDataset,
    MammographyScreener,
    VIEW_KEYS,
    build_exam_records,
    collate_batch,
    filter_valid_exams,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Processed VinDr directory containing metadata.csv")
    parser.add_argument("--checkpoint", required=True, help="Legacy best_model.pt checkpoint path")
    parser.add_argument(
        "--output-dir",
        default="reports/paper2/stage1_statistics",
        help="Directory for predictions.json and stage1_statistical_depth.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_iterations: int, seed: int) -> tuple[float, float]:
    values: list[float] = []
    n = len(y_true)
    rng = np.random.RandomState(seed)
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            values.append(float(metric_fn(yt, ys)))
        except Exception:
            continue
    if not values:
        return 0.0, 0.0
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return lower, upper


def calibration_bins(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    for idx in range(n_bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_score >= low) & (y_score <= high)
        else:
            mask = (y_score >= low) & (y_score < high)
        if not np.any(mask):
            continue
        bins.append(
            {
                "bin": idx + 1,
                "lower": round(float(low), 4),
                "upper": round(float(high), 4),
                "mean_predicted_probability": round(float(y_score[mask].mean()), 4),
                "observed_positive_rate": round(float(y_true[mask].mean()), 4),
                "n_exams": int(mask.sum()),
            }
        )
    return bins


def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.90) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs((1 - fpr) - target_spec)))
    return float(tpr[idx]), float(thresholds[idx])


def specificity_at_sensitivity(y_true: np.ndarray, y_score: np.ndarray, target_sens: float = 0.90) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(tpr - target_sens)))
    return float(1 - fpr[idx]), float(thresholds[idx])


def run_inference(args: argparse.Namespace) -> tuple[list[dict], dict]:
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    raw_dir = data_dir.parent / "raw"
    exams = build_exam_records(metadata_path, raw_dir)
    if not exams:
        raise RuntimeError(f"No complete 4-view exams found under {raw_dir}")
    exams, dropped = filter_valid_exams(exams)
    test_exams = [exam for exam in exams if exam["split"] == "test"]
    if not test_exams:
        raise RuntimeError("No valid test exams available after DICOM validation")

    test_ds = MammographyExamDataset(test_exams, args.image_size)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    # Avoid any remote weight downloads; the checkpoint provides the trained weights.
    model = MammographyScreener(pretrained=False)
    if model.encoder.backbone is None:
        raise RuntimeError("timm is required for mammography inference but is not installed")
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    predictions: list[dict] = []
    with torch.no_grad():
        for batch in test_loader:
            views = {key: value.to(device) for key, value in batch["views"].items()}
            labels = batch["labels"].detach().cpu().numpy()
            logits, _ = model(views)
            probs = torch.sigmoid(logits.squeeze(1)).detach().cpu().numpy()
            for study_id, label, prob in zip(batch["study_ids"], labels.tolist(), probs.tolist()):
                predictions.append(
                    {
                        "study_id": study_id,
                        "true_label": int(label),
                        "predicted_probability": float(prob),
                    }
                )

    manifest = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "n_test_exams": len(test_ds),
        "dropped_invalid_exams": int(dropped),
        "image_size": args.image_size,
        "views_required": VIEW_KEYS,
    }
    return predictions, manifest


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, manifest = run_inference(args)
    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(json.dumps(predictions, indent=2))

    y_true = np.asarray([row["true_label"] for row in predictions], dtype=np.int64)
    y_score = np.asarray([row["predicted_probability"] for row in predictions], dtype=np.float64)

    auroc_point = float(roc_auc_score(y_true, y_score))
    sens90_point, sens90_threshold = sensitivity_at_specificity(y_true, y_score, 0.90)
    spec90_point, spec90_threshold = specificity_at_sensitivity(y_true, y_score, 0.90)
    auroc_ci = bootstrap_ci(y_true, y_score, roc_auc_score, args.bootstrap_iterations, args.seed)
    sens_ci = bootstrap_ci(
        y_true,
        y_score,
        lambda yt, ys: sensitivity_at_specificity(yt, ys, 0.90)[0],
        args.bootstrap_iterations,
        args.seed,
    )
    spec_ci = bootstrap_ci(
        y_true,
        y_score,
        lambda yt, ys: specificity_at_sensitivity(yt, ys, 0.90)[0],
        args.bootstrap_iterations,
        args.seed,
    )

    brier = float(brier_score_loss(y_true, y_score))
    calibration = calibration_bins(y_true, y_score, n_bins=10)

    youden_fpr, youden_tpr, youden_thresholds = roc_curve(y_true, y_score)
    youden_idx = int(np.argmax(youden_tpr - youden_fpr))
    youden_threshold = float(youden_thresholds[youden_idx])

    y_pred_90spec = (y_score >= sens90_threshold).astype(np.int64)
    y_pred_youden = (y_score >= youden_threshold).astype(np.int64)
    cm_90spec = confusion_matrix(y_true, y_pred_90spec)
    cm_youden = confusion_matrix(y_true, y_pred_youden)

    false_negatives_90spec = [
        row["study_id"] for row, pred in zip(predictions, y_pred_90spec.tolist()) if row["true_label"] == 1 and pred == 0
    ]
    false_positives_90spec = [
        row["study_id"] for row, pred in zip(predictions, y_pred_90spec.tolist()) if row["true_label"] == 0 and pred == 1
    ]

    summary = {
        "manifest": manifest,
        "n_test": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": round(float(y_true.mean()), 4),
        "bootstrap_iterations": args.bootstrap_iterations,
        "auroc": {
            "point": round(auroc_point, 4),
            "ci_lower": round(auroc_ci[0], 4),
            "ci_upper": round(auroc_ci[1], 4),
        },
        "sensitivity_at_90_specificity": {
            "point": round(sens90_point, 4),
            "threshold": round(sens90_threshold, 6),
            "ci_lower": round(sens_ci[0], 4),
            "ci_upper": round(sens_ci[1], 4),
        },
        "specificity_at_90_sensitivity": {
            "point": round(spec90_point, 4),
            "threshold": round(spec90_threshold, 6),
            "ci_lower": round(spec_ci[0], 4),
            "ci_upper": round(spec_ci[1], 4),
        },
        "brier_score": round(brier, 4),
        "calibration_curve": calibration,
        "confusion_matrix_90spec": {
            "threshold": round(sens90_threshold, 6),
            "tn": int(cm_90spec[0, 0]),
            "fp": int(cm_90spec[0, 1]),
            "fn": int(cm_90spec[1, 0]),
            "tp": int(cm_90spec[1, 1]),
            "false_negative_study_ids": false_negatives_90spec,
            "false_positive_study_ids": false_positives_90spec,
        },
        "confusion_matrix_youden": {
            "threshold": round(youden_threshold, 6),
            "tn": int(cm_youden[0, 0]),
            "fp": int(cm_youden[0, 1]),
            "fn": int(cm_youden[1, 0]),
            "tp": int(cm_youden[1, 1]),
        },
    }
    (output_dir / "stage1_statistical_depth.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
