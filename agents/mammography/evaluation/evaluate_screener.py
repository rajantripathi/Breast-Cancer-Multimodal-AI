from __future__ import annotations

"""Evaluate mammography screening predictions or a legacy checkpoint."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from evaluation.statistics import (
    binary_confusion_at_threshold,
    binary_auroc,
    bootstrap_confidence_interval,
    calibration_bins,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)
from training.reproducibility import build_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mammography screening predictions or a legacy checkpoint")
    parser.add_argument("--model-path", default=None, help="Checkpoint path. Used only when --predictions-json is absent.")
    parser.add_argument("--data-dir", default=None, help="Processed dataset directory containing metadata.csv.")
    parser.add_argument("--metadata-csv", default=None, help="Optional metadata CSV override.")
    parser.add_argument("--raw-dir", default=None, help="Optional raw image root override for checkpoint inference.")
    parser.add_argument("--predictions-json", default=None, help="Existing predictions JSON to summarize.")
    parser.add_argument("--output-dir", default="reports/mammography")
    parser.add_argument("--model-type", choices=["auto", "legacy", "standard"], default="auto")
    parser.add_argument(
        "--eval-split",
        choices=["auto", "all", "train", "val", "test", "external"],
        default="auto",
        help="Which split to evaluate when running checkpoint inference.",
    )
    parser.add_argument(
        "--harmonization-stats-json",
        default=None,
        help="Optional train-fit harmonization stats JSON for non-legacy checkpoint inference.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalize_predictions(raw_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in raw_predictions:
        label = row.get("true_label", 0)
        if isinstance(label, str):
            label = 1 if label.strip().lower() in {"1", "true", "high_concern", "positive"} else 0
        score = row.get("predicted_probability", row.get("risk_score"))
        if score is None and isinstance(row.get("probabilities"), dict):
            score = row["probabilities"].get("high_concern")
        if score is None:
            raise ValueError(f"Prediction row missing score field: {row}")
        study_id = str(row.get("study_id") or row.get("sample_id") or "")
        if not study_id:
            raise ValueError(f"Prediction row missing study identifier: {row}")
        predictions.append(
            {
                "study_id": study_id,
                "dataset_source": str(row.get("dataset_source") or "unknown"),
                "true_label": int(label),
                "predicted_probability": float(score),
            }
        )
    return predictions


def _resolve_metadata_path(args: argparse.Namespace) -> Path:
    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv)
    elif args.data_dir:
        metadata_path = Path(args.data_dir) / "metadata.csv"
    else:
        raise ValueError("--metadata-csv or --data-dir is required for checkpoint inference")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return metadata_path


def _resolve_raw_dir(args: argparse.Namespace, metadata_path: Path) -> Path:
    if args.raw_dir:
        return Path(args.raw_dir)
    if args.data_dir:
        return Path(args.data_dir).parent / "raw"
    return metadata_path.parent.parent / "raw"


def _select_eval_exams(exams: list[dict[str, Any]], eval_split: str) -> list[dict[str, Any]]:
    if eval_split == "all":
        return exams
    if eval_split != "auto":
        return [exam for exam in exams if str(exam.get("split", "")).strip().lower() == eval_split]

    explicit_priority = ["test", "external", "val", "train"]
    for split_name in explicit_priority:
        subset = [exam for exam in exams if str(exam.get("split", "")).strip().lower() == split_name]
        if subset:
            return subset
    return exams


def _infer_model_type_from_checkpoint(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        return "legacy"
    checkpoint_args = checkpoint.get("args")
    if isinstance(checkpoint_args, dict):
        if any(key in checkpoint_args for key in ("effective_batch_size", "tta", "loss", "aux_metadata_csv")):
            return "standard"
    return "legacy"


def _run_legacy_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    from agents.mammography.training.train_screener_legacy import (
        MammographyExamDataset,
        MammographyScreener,
        build_exam_records,
        collate_batch,
        filter_valid_exams,
        resolve_device,
        set_seed,
    )

    @torch.no_grad()
    def _infer() -> list[dict[str, Any]]:
        if not args.model_path:
            raise ValueError("--model-path is required when --predictions-json is not provided")
        set_seed(int(args.seed))
        device = resolve_device(args.device)
        metadata_path = _resolve_metadata_path(args)
        raw_dir = _resolve_raw_dir(args, metadata_path)
        exams = build_exam_records(metadata_path, raw_dir)
        if not exams:
            raise RuntimeError(f"No complete 4-view exams found under {raw_dir}")
        exams, _ = filter_valid_exams(exams)
        test_exams = _select_eval_exams(exams, str(args.eval_split))
        if not test_exams:
            raise RuntimeError(f"No valid exams available for eval_split={args.eval_split}")

        dataset = MammographyExamDataset(test_exams, int(args.image_size))
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            collate_fn=collate_batch,
        )

        model = MammographyScreener(pretrained=False)
        if model.encoder.backbone is None:
            raise RuntimeError("timm is required for mammography inference but is not installed")
        checkpoint = torch.load(Path(args.model_path), map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        predictions: list[dict[str, Any]] = []
        for batch in loader:
            views = {key: value.to(device) for key, value in batch["views"].items()}
            logits, _ = model(views)
            probabilities = torch.sigmoid(logits.squeeze(1)).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            for study_id, label, probability in zip(batch["study_ids"], labels.tolist(), probabilities.tolist()):
                predictions.append(
                    {
                        "study_id": str(study_id),
                        "dataset_source": "vindr",
                        "true_label": int(label),
                        "predicted_probability": float(probability),
                    }
                )
        return predictions

    return _infer()


def _run_standard_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    from agents.mammography.preprocessing.harmonization import load_harmonization_stats
    from agents.mammography.training.train_screener import (
        MammographyExamDataset,
        MammographyScreener,
        build_exam_records,
        collate_batch,
        filter_valid_exams,
        resolve_device,
        set_seed,
    )

    @torch.no_grad()
    def _infer() -> list[dict[str, Any]]:
        if not args.model_path:
            raise ValueError("--model-path is required when --predictions-json is not provided")
        set_seed(int(args.seed))
        device = resolve_device(args.device)
        metadata_path = _resolve_metadata_path(args)
        raw_dir = _resolve_raw_dir(args, metadata_path)
        exams = build_exam_records([(metadata_path, raw_dir, False)])
        if not exams:
            raise RuntimeError(f"No complete 4-view exams found under metadata={metadata_path}")
        exams, _ = filter_valid_exams(exams)
        eval_exams = _select_eval_exams(exams, str(args.eval_split))
        if not eval_exams:
            raise RuntimeError(f"No valid exams available for eval_split={args.eval_split}")

        harmonization_stats = None
        if args.harmonization_stats_json:
            harmonization_stats = load_harmonization_stats(args.harmonization_stats_json)
        dataset = MammographyExamDataset(
            eval_exams,
            int(args.image_size),
            training=False,
            harmonization_stats=harmonization_stats,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            collate_fn=collate_batch,
        )

        model = MammographyScreener(pretrained=True)
        if model.encoder.backbone is None:
            raise RuntimeError("timm is required for mammography inference but is not installed")
        checkpoint = torch.load(Path(args.model_path), map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        predictions: list[dict[str, Any]] = []
        for batch in loader:
            views = {key: value.to(device) for key, value in batch["views"].items()}
            logits, _ = model(views)
            probabilities = torch.sigmoid(logits.squeeze(1)).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            for study_id, dataset_source, label, probability in zip(
                batch["study_ids"],
                batch["dataset_sources"],
                labels.tolist(),
                probabilities.tolist(),
            ):
                predictions.append(
                    {
                        "study_id": str(study_id),
                        "dataset_source": str(dataset_source),
                        "true_label": int(label),
                        "predicted_probability": float(probability),
                    }
                )
        return predictions

    return _infer()


def _density_subgroups(predictions: list[dict[str, Any]], metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    import pandas as pd

    metadata = pd.read_csv(metadata_path)
    if "exam_density" not in metadata.columns:
        return {}
    density_lookup = (
        metadata.loc[:, ["study_id", "exam_density"]]
        .drop_duplicates("study_id")
        .assign(exam_density=lambda frame: frame["exam_density"].fillna("").astype(str).str.strip().str.upper())
    )
    density_map = dict(zip(density_lookup["study_id"].astype(str), density_lookup["exam_density"]))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        raw_study_id = str(row["study_id"]).split(":", 1)[-1]
        density = density_map.get(raw_study_id, "") or "UNKNOWN"
        grouped.setdefault(density, []).append(row)

    summary: dict[str, Any] = {}
    for density, rows in sorted(grouped.items()):
        labels = [int(row["true_label"]) for row in rows]
        scores = [float(row["predicted_probability"]) for row in rows]
        positives = int(sum(labels))
        summary[density] = {
            "n_exams": len(rows),
            "n_positive": positives,
            "prevalence": round(positives / len(rows), 4) if rows else 0.0,
            "auroc": round(binary_auroc(labels, scores), 4) if len(set(labels)) > 1 else None,
        }
    return summary


def _source_subgroups(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        source = str(row.get("dataset_source") or "unknown").strip().lower() or "unknown"
        grouped.setdefault(source, []).append(row)

    summary: dict[str, Any] = {}
    for source, rows in sorted(grouped.items()):
        labels = [int(row["true_label"]) for row in rows]
        scores = [float(row["predicted_probability"]) for row in rows]
        positives = int(sum(labels))
        summary[source] = {
            "n_exams": len(rows),
            "n_positive": positives,
            "prevalence": round(positives / len(rows), 4) if rows else 0.0,
            "auroc": round(binary_auroc(labels, scores), 4) if len(set(labels)) > 1 else None,
        }
    return summary


def _summarize_predictions(
    predictions: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
    metadata_path: Path | None,
) -> dict[str, Any]:
    labels = [int(row["true_label"]) for row in predictions]
    scores = [float(row["predicted_probability"]) for row in predictions]
    if not labels:
        raise ValueError("No predictions available for evaluation")

    auroc = binary_auroc(labels, scores)
    auroc_ci = bootstrap_confidence_interval(
        lambda sample_labels, sample_scores: binary_auroc(sample_labels, sample_scores),
        labels,
        scores,
        iterations=bootstrap_iterations,
        seed=seed,
        stratify=True,
    )
    sens90, sens90_threshold = sensitivity_at_specificity(labels, scores, 0.90)
    sens90_ci = bootstrap_confidence_interval(
        lambda sample_labels, sample_scores: sensitivity_at_specificity(sample_labels, sample_scores, 0.90)[0],
        labels,
        scores,
        iterations=bootstrap_iterations,
        seed=seed,
        stratify=True,
    )
    spec90, spec90_threshold = specificity_at_sensitivity(labels, scores, 0.90)
    spec90_ci = bootstrap_confidence_interval(
        lambda sample_labels, sample_scores: specificity_at_sensitivity(sample_labels, sample_scores, 0.90)[0],
        labels,
        scores,
        iterations=bootstrap_iterations,
        seed=seed,
        stratify=True,
    )
    brier = float(np.mean((np.asarray(labels, dtype=np.float64) - np.asarray(scores, dtype=np.float64)) ** 2))
    confusion_90spec = binary_confusion_at_threshold(labels, scores, sens90_threshold)
    threshold_grid = np.unique(np.asarray(scores, dtype=np.float64))
    youden_scores = []
    for threshold in threshold_grid:
        confusion = binary_confusion_at_threshold(labels, scores, float(threshold))
        tp = int(confusion["tp"])
        tn = int(confusion["tn"])
        fp = int(confusion["fp"])
        fn = int(confusion["fn"])
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        youden_scores.append((sensitivity + specificity - 1.0, float(threshold)))
    _, youden_threshold = max(youden_scores, key=lambda item: item[0])
    confusion_youden = binary_confusion_at_threshold(labels, scores, youden_threshold)
    density_summary = _density_subgroups(predictions, metadata_path)

    return {
        "n_test": len(predictions),
        "n_positive": int(sum(labels)),
        "prevalence": round(float(sum(labels) / len(labels)), 4),
        "bootstrap_iterations": int(bootstrap_iterations),
        "auroc": {
            "point": round(float(auroc), 4),
            "ci_lower": round(float(auroc_ci[0]), 4),
            "ci_upper": round(float(auroc_ci[1]), 4),
        },
        "sensitivity_at_90_specificity": {
            "point": round(float(sens90), 4),
            "threshold": round(float(sens90_threshold), 6),
            "ci_lower": round(float(sens90_ci[0]), 4),
            "ci_upper": round(float(sens90_ci[1]), 4),
        },
        "specificity_at_90_sensitivity": {
            "point": round(float(spec90), 4),
            "threshold": round(float(spec90_threshold), 6),
            "ci_lower": round(float(spec90_ci[0]), 4),
            "ci_upper": round(float(spec90_ci[1]), 4),
        },
        "brier_score": round(float(brier), 4),
        "calibration_curve": calibration_bins(labels, scores, num_bins=10),
        "confusion_matrix_90spec": confusion_90spec,
        "confusion_matrix_youden": confusion_youden,
        "source_subgroups": _source_subgroups(predictions),
        "density_subgroups": density_summary,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = None
    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv)
    elif args.data_dir:
        candidate = Path(args.data_dir) / "metadata.csv"
        if candidate.exists():
            metadata_path = candidate

    if args.predictions_json:
        raw_predictions = json.loads(Path(args.predictions_json).read_text())
        predictions = _normalize_predictions(raw_predictions)
        input_paths = [Path(args.predictions_json)]
        if metadata_path is not None:
            input_paths.append(metadata_path)
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when --predictions-json is not provided")
        selected_model_type = str(args.model_type)
        if selected_model_type == "auto":
            import torch

            checkpoint = torch.load(Path(args.model_path), map_location="cpu")
            selected_model_type = _infer_model_type_from_checkpoint(checkpoint)
        if metadata_path is None:
            metadata_path = _resolve_metadata_path(args)
        if selected_model_type == "standard":
            predictions = _run_standard_inference(args)
        else:
            predictions = _run_legacy_inference(args)
        input_paths = [Path(args.model_path), metadata_path]
        if args.harmonization_stats_json:
            input_paths.append(Path(args.harmonization_stats_json))

    summary = _summarize_predictions(
        predictions,
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
        metadata_path=metadata_path,
    )
    manifest = build_run_manifest(
        task="mammography_screener_evaluation",
        args=args,
        input_paths=[path for path in input_paths if path is not None],
        split_counts={"n_predictions": len(predictions)},
        seed_state={"seed": int(args.seed), "bootstrap_iterations": int(args.bootstrap_iterations)},
        extra={
            "output_dir": str(output_dir),
            "model_type": args.model_type if args.predictions_json else selected_model_type,
            "eval_split": str(args.eval_split),
        },
        repo_root=Path(__file__).resolve().parents[3],
    )

    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
