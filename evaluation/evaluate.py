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
    labels = []
    for preferred in ("monitor", "high_concern"):
        if any(item.get("true_label") == preferred or item.get("predicted_label") == preferred for item in predictions):
            labels.append(preferred)
    for candidate in sorted({item["true_label"] for item in predictions} | {item["predicted_label"] for item in predictions}):
        if candidate not in labels:
            labels.append(candidate)
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = [label_to_index[item["true_label"]] for item in predictions]
    y_pred = []
    y_score = []
    confidences = []
    for item in predictions:
        risk_score = item.get("risk_score")
        if len(labels) == 2 and risk_score is not None and {"monitor", "high_concern"} <= set(labels):
            risk_value = min(max(float(risk_score), 0.0), 1.0)
            probability_map = {"monitor": 1.0 - risk_value, "high_concern": risk_value}
            probability_row = [probability_map.get(label, 0.0) for label in labels]
            predicted_label = "high_concern" if risk_value >= 0.5 else "monitor"
        else:
            probability_row = [float(item.get("probabilities", {}).get(label, 0.0)) for label in labels]
            total = sum(probability_row)
            probability_row = [value / total for value in probability_row] if total > 0 else [1.0 / len(labels)] * len(labels)
            predicted_label = item["predicted_label"]
        y_pred.append(label_to_index.get(predicted_label, label_to_index[item["predicted_label"]]))
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
        positive_label = "high_concern" if "high_concern" in labels else labels[-1]
        positive_index = labels.index(positive_label)
        positive_scores = [row[positive_index] for row in y_score]
        metrics["auroc_macro"] = round(_binary_auroc(y_true, positive_scores), 4)
        metrics["auprc_macro"] = round(_binary_auprc(y_true, positive_scores), 4)
        metrics["brier_score"] = round(sum((true - score) ** 2 for true, score in zip(y_true, positive_scores)) / len(y_true), 4)
        auroc_ci = bootstrap_metric(lambda yt, ys: _binary_auroc(yt, [row[positive_index] for row in ys]), y_true, y_pred, y_score)
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


def _risk_group_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {}
    groups = {
        "low_risk": {"count": 0, "events": 0, "mean_risk_score": 0.0},
        "intermediate_risk": {"count": 0, "events": 0, "mean_risk_score": 0.0},
        "high_risk": {"count": 0, "events": 0, "mean_risk_score": 0.0},
    }
    for item in predictions:
        score = float(item.get("risk_score", item.get("probabilities", {}).get("high_concern", 0.0)))
        if score >= 0.67:
            bucket = "high_risk"
        elif score >= 0.33:
            bucket = "intermediate_risk"
        else:
            bucket = "low_risk"
        groups[bucket]["count"] += 1
        groups[bucket]["events"] += int(item.get("event_observed", 0))
        groups[bucket]["mean_risk_score"] += score
    for summary in groups.values():
        count = summary["count"]
        summary["mean_risk_score"] = round(summary["mean_risk_score"] / count, 4) if count else 0.0
        summary["event_rate"] = round(summary["events"] / count, 4) if count else 0.0
    return groups


def _modality_agreement_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {}
    summary: dict[str, Any] = {}
    fused_labels = [str(item.get("predicted_label", "")) for item in predictions]
    for modality in ("vision", "clinical", "genomics"):
        modality_labels = [
            str(item.get("modality_predictions", {}).get(modality, {}).get("class", ""))
            for item in predictions
        ]
        matches = sum(1 for fused, modality_label in zip(fused_labels, modality_labels) if fused and fused == modality_label)
        high_concern = sum(1 for label in modality_labels if label == "high_concern")
        summary[modality] = {
            "agreement_with_fused": round(matches / len(predictions), 4),
            "high_concern_rate": round(high_concern / len(predictions), 4),
        }
    return summary


def _harrell_c_index(survival_times: list[float], risk_scores: list[float], event_observed: list[int]) -> float:
    concordant = 0.0
    admissible = 0
    total = len(survival_times)
    for i in range(total):
        for j in range(i + 1, total):
            t_i, t_j = survival_times[i], survival_times[j]
            e_i, e_j = event_observed[i], event_observed[j]
            r_i, r_j = risk_scores[i], risk_scores[j]
            if t_i == t_j and not (e_i or e_j):
                continue
            if t_i < t_j and e_i:
                admissible += 1
                if r_i > r_j:
                    concordant += 1.0
                elif r_i == r_j:
                    concordant += 0.5
            elif t_j < t_i and e_j:
                admissible += 1
                if r_j > r_i:
                    concordant += 1.0
                elif r_i == r_j:
                    concordant += 0.5
        # pairs with identical event time and both observed are intentionally skipped here
    if admissible == 0:
        raise ZeroDivisionError
    return concordant / admissible


def _survival_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions or not all("survival_time" in item and "event_observed" in item for item in predictions):
        return {"c_index_message": "Survival labels not available; C-index skipped"}
    survival_times = [float(item["survival_time"]) for item in predictions]
    event_observed = [int(item["event_observed"]) for item in predictions]
    risk_scores = [float(item.get("risk_score", item.get("probabilities", {}).get("high_concern", 0.0))) for item in predictions]
    survival_min = min(survival_times) if survival_times else 0.0
    survival_max = max(survival_times) if survival_times else 0.0
    survival_unique = len(set(survival_times))
    event_sum = sum(event_observed)
    print(
        f"Survival times: min={survival_min}, max={survival_max}, unique={survival_unique}",
        flush=True,
    )
    print(f"Events: sum={event_sum}, total={len(event_observed)}", flush=True)
    try:
        try:
            from lifelines.utils import concordance_index

            c_index = float(concordance_index(survival_times, risk_scores, event_observed))
        except ImportError:
            c_index = float(_harrell_c_index(survival_times, risk_scores, event_observed))
        return {
            "c_index": round(c_index, 4),
            "survival_time_diagnostic": {
                "min": survival_min,
                "max": survival_max,
                "unique": survival_unique,
            },
            "event_diagnostic": {
                "sum": event_sum,
                "total": len(event_observed),
            },
        }
    except ZeroDivisionError:
        return {
            "c_index_message": "Survival labels present but no admissible pairs; C-index skipped",
            "survival_time_diagnostic": {
                "min": survival_min,
                "max": survival_max,
                "unique": survival_unique,
            },
            "event_diagnostic": {
                "sum": event_sum,
                "total": len(event_observed),
            },
        }


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
    fused_labels = [
        item.get("risk_classification", item.get("fused_label", item.get("predicted_label", "unknown")))
        for item in verifier_predictions
    ]
    alignment_summary = _alignment_summary(verifier_artifact)
    classification_metrics = _classification_metrics(verifier_predictions)
    survival_metrics = _survival_metrics(verifier_predictions)

    metrics: dict[str, object] = {
        "num_predictions": len(predictions),
        "fused_label_distribution": label_distribution(fused_labels),
        "alignment_summary": alignment_summary,
    }
    metrics.update({key: value for key, value in classification_metrics.items() if key not in {"classification_report", "confusion_matrix", "labels"}})
    metrics["risk_group_summary"] = _risk_group_summary(verifier_predictions)
    metrics["modality_agreement_summary"] = _modality_agreement_summary(verifier_predictions)
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
