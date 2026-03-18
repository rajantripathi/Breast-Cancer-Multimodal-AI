from __future__ import annotations

"""Enterprise-style evaluation for verifier and sample-case outputs."""

import argparse
import json
from pathlib import Path
from typing import Any

from .metrics import bootstrap_metric, expected_calibration_error, label_distribution
from .visualize import render_text_report


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _alignment_summary(verifier_artifact: dict[str, Any] | None) -> str:
    if not verifier_artifact:
        return "Verifier alignment status unavailable"
    status = verifier_artifact.get("alignment_status", "unaligned_legacy")
    count = int(verifier_artifact.get("aligned_sample_count", 0))
    if status in {"patient_aligned", "patient_aligned_tcga"}:
        return f"Verifier trained on {count} patient-aligned bundles"
    return "Verifier trained on unaligned bundles (legacy mode)"


def _enrich_survival_from_clinical(
    predictions: list[dict[str, Any]],
    clinical_csv: str | Path | None,
) -> list[dict[str, Any]]:
    if not predictions or clinical_csv is None:
        return predictions
    if all("survival_time" in item and "event_observed" in item for item in predictions):
        return predictions
    clinical_path = Path(clinical_csv)
    if not clinical_path.exists():
        return predictions
    import pandas as pd

    frame = pd.read_csv(clinical_path).copy()
    if "bcr_patient_barcode" not in frame.columns:
        return predictions
    frame["patient_barcode"] = frame["bcr_patient_barcode"].astype(str).str.upper().str.slice(0, 12)
    lookup = frame.set_index("patient_barcode").to_dict(orient="index")
    enriched: list[dict[str, Any]] = []
    for item in predictions:
        clone = dict(item)
        sample_id = str(clone.get("sample_id", "")).upper()[:12]
        row = lookup.get(sample_id)
        if row is not None:
            if "survival_time" not in clone:
                survival_time = row.get("days_to_death") if pd.notna(row.get("days_to_death")) else row.get("days_to_last_followup")
                clone["survival_time"] = float(survival_time) if pd.notna(survival_time) else 0.0
            if "event_observed" not in clone:
                vital_status = str(row.get("vital_status", "")).strip().lower()
                clone["event_observed"] = int(vital_status in {"dead", "deceased", "1", "true", "yes"})
        enriched.append(clone)
    return enriched


def _extract_prediction_arrays(predictions: list[dict[str, Any]]) -> tuple[list[str], list[int], list[int], list[list[float]], list[float]]:
    labels = sorted({item["true_label"] for item in predictions} | {item["predicted_label"] for item in predictions})
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = [label_to_index[item["true_label"]] for item in predictions]
    y_pred = [label_to_index[item["predicted_label"]] for item in predictions]
    y_score = []
    confidences = []
    for item in predictions:
        probability_row = [float(item.get("probabilities", {}).get(label, 0.0)) for label in labels]
        total = sum(probability_row)
        probability_row = [value / total for value in probability_row] if total > 0 else [1.0 / len(labels)] * len(labels)
        y_score.append(probability_row)
        confidences.append(max(probability_row))
    return labels, y_true, y_pred, y_score, confidences


def _balanced_accuracy(y_true: list[int], y_pred: list[int], num_labels: int) -> float:
    recalls = []
    for label in range(num_labels):
        positives = [index for index, true in enumerate(y_true) if true == label]
        if not positives:
            continue
        recalls.append(sum(1 for index in positives if y_pred[index] == label) / len(positives))
    return sum(recalls) / len(recalls) if recalls else 0.0


def _f1_macro(y_true: list[int], y_pred: list[int], num_labels: int) -> float:
    scores = []
    for label in range(num_labels):
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _confusion_matrix(y_true: list[int], y_pred: list[int], num_labels: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix


def _binary_auroc(y_true: list[int], positive_scores: list[float]) -> float:
    positives = [(score, true) for score, true in zip(positive_scores, y_true) if true == 1]
    negatives = [(score, true) for score, true in zip(positive_scores, y_true) if true == 0]
    if not positives or not negatives:
        return 0.0
    concordant = 0.0
    total = 0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            total += 1
            if pos_score > neg_score:
                concordant += 1.0
            elif pos_score == neg_score:
                concordant += 0.5
    return concordant / total if total else 0.0


def _binary_auprc(y_true: list[int], positive_scores: list[float]) -> float:
    ranked = sorted(zip(positive_scores, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(y_true)
    if positives == 0:
        return 0.0
    tp = 0
    fp = 0
    area = 0.0
    prev_recall = 0.0
    for score, label in ranked:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def _classification_report(labels: list[str], matrix: list[list[int]]) -> str:
    lines = ["label,precision,recall,f1,support"]
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(labels)) if row != index)
        fn = sum(matrix[index][col] for col in range(len(labels)) if col != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        lines.append(f"{label},{precision:.4f},{recall:.4f},{f1:.4f},{support}")
    return "\n".join(lines)


def _classification_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {}
    labels, y_true, y_pred, y_score, confidences = _extract_prediction_arrays(predictions)
    num_labels = len(labels)
    matrix = _confusion_matrix(y_true, y_pred, num_labels)
    metrics: dict[str, Any] = {
        "balanced_accuracy": round(_balanced_accuracy(y_true, y_pred, num_labels), 4),
        "f1_macro": round(_f1_macro(y_true, y_pred, num_labels), 4),
        "ece": round(expected_calibration_error(y_true, confidences, y_pred), 4),
        "confusion_matrix": matrix,
        "classification_report": _classification_report(labels, matrix),
        "labels": labels,
    }
    if num_labels == 2:
        positive_scores = [row[1] for row in y_score]
        metrics["auroc_macro"] = round(_binary_auroc(y_true, positive_scores), 4)
        metrics["auprc_macro"] = round(_binary_auprc(y_true, positive_scores), 4)
        metrics["brier_score"] = round(sum((true - score) ** 2 for true, score in zip(y_true, positive_scores)) / len(y_true), 4)
        auroc_ci = bootstrap_metric(lambda yt, ys: _binary_auroc(yt, [row[1] for row in ys]), y_true, y_pred, y_score)
    else:
        metrics["auroc_macro"] = 0.0
        metrics["auprc_macro"] = 0.0
        metrics["brier_score"] = round(
            sum((1.0 - y_score[index][true]) ** 2 for index, true in enumerate(y_true)) / len(y_true),
            4,
        )
        auroc_ci = (0.0, 0.0)
    bal_ci = bootstrap_metric(lambda yt, yp: _balanced_accuracy(yt, yp, num_labels), y_true, y_pred)
    metrics["auroc_ci_95"] = [round(auroc_ci[0], 4), round(auroc_ci[1], 4)]
    metrics["balanced_accuracy_ci_95"] = [round(bal_ci[0], 4), round(bal_ci[1], 4)]
    return metrics


def _survival_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions or not all("survival_time" in item and "event_observed" in item for item in predictions):
        return {"c_index_message": "Survival labels not available; C-index skipped"}
    try:
        from lifelines.utils import concordance_index
    except ImportError:
        return {"c_index_message": "Survival labels not available; C-index skipped"}
    survival_times = [float(item["survival_time"]) for item in predictions]
    event_observed = [int(item["event_observed"]) for item in predictions]
    risk_scores = [max(item.get("probabilities", {}).values()) if item.get("probabilities") else 0.0 for item in predictions]
    try:
        return {"c_index": round(float(concordance_index(survival_times, risk_scores, event_observed)), 4)}
    except ZeroDivisionError:
        return {"c_index_message": "Survival labels present but no admissible pairs; C-index skipped"}


def evaluate_predictions(
    prediction_file: str | Path,
    verifier_artifact_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    clinical_csv: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate current outputs and write enterprise-style reports."""
    prediction_path = Path(prediction_file)
    repo_root = prediction_path.resolve().parents[1] if prediction_path.exists() else Path(__file__).resolve().parents[1]
    predictions = _load_json(prediction_path) or []
    artifact_path = Path(verifier_artifact_path) if verifier_artifact_path else (repo_root / "outputs" / "verifier" / "artifact.json")
    verifier_artifact = _load_json(artifact_path)
    verifier_predictions = predictions if predictions else (verifier_artifact.get("predictions", []) if verifier_artifact else [])
    verifier_predictions = _enrich_survival_from_clinical(verifier_predictions, clinical_csv)
    fused_labels = [item.get("risk_classification", item.get("fused_label", "unknown")) for item in predictions]
    alignment_summary = _alignment_summary(verifier_artifact)
    classification_metrics = _classification_metrics(verifier_predictions)
    survival_metrics = _survival_metrics(verifier_predictions)

    metrics: dict[str, object] = {
        "num_predictions": len(predictions),
        "fused_label_distribution": label_distribution(fused_labels),
        "alignment_summary": alignment_summary,
    }
    metrics.update({key: value for key, value in classification_metrics.items() if key not in {"classification_report", "confusion_matrix", "labels"}})
    metrics.update(survival_metrics)

    outputs_root = Path(output_dir) if output_dir else prediction_path.parent
    outputs_root.mkdir(parents=True, exist_ok=True)
    labels = classification_metrics.get("labels", [])
    confusion = classification_metrics.get("confusion_matrix")
    if confusion and labels:
        confusion_csv = outputs_root / "confusion_matrix.csv"
        rows = [",".join(["label"] + [str(label) for label in labels])]
        for label, values in zip(labels, confusion):
            rows.append(",".join([str(label)] + [str(value) for value in values]))
        confusion_csv.write_text("\n".join(rows) + "\n")
    if classification_metrics.get("classification_report"):
        (outputs_root / "classification_report.txt").write_text(str(classification_metrics["classification_report"]))

    summary_json = outputs_root / "enterprise_metrics.json"
    summary_json.write_text(json.dumps(metrics, indent=2))
    report_path = outputs_root / "evaluation_report.txt"
    report_path.write_text(render_text_report(metrics))
    return metrics


def main() -> None:
    """CLI entrypoint for enterprise evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate verifier outputs")
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--clinical-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
        prediction_file = experiment_dir / "predictions.json"
        artifact_file = experiment_dir / "artifact.json"
        output_dir = Path(args.output_dir or experiment_dir)
        metrics = evaluate_predictions(
            prediction_file,
            verifier_artifact_path=artifact_file,
            output_dir=output_dir,
            clinical_csv=args.clinical_csv,
        )
    else:
        sample_file = Path(__file__).resolve().parents[1] / "outputs" / "sample_case_results.json"
        if not sample_file.exists():
            sample_file = Path(__file__).resolve().parents[1] / "outputs" / "fused_predictions.json"
        if not sample_file.exists():
            sample_file.write_text("[]")
        metrics = evaluate_predictions(sample_file)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
