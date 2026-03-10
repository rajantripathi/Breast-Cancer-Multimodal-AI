from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents import EHRAgent, GenomicsAgent, LiteratureAgent, VisionAgent
from config import load_settings
from data.common import flatten_payload, read_json
from training.utils import score_with_prototypes

POSITIVE_LABELS = {
    "vision": {"malignant"},
    "ehr": {"high_risk"},
    "genomics": {"pathogenic_variant"},
    "literature": {"supportive_evidence"},
}
MODALITY_WEIGHTS = {"vision": 1.2, "ehr": 1.0, "genomics": 1.35, "literature": 0.55}


def _artifact_path(repo_root: Path, modality: str) -> str:
    return str(repo_root / "outputs" / modality / "artifact.json")


def _load_verifier(repo_root: Path) -> dict[str, Any]:
    artifact_path = repo_root / "outputs" / "verifier" / "artifact.json"
    if artifact_path.exists():
        return read_json(artifact_path)
    return {"prototypes": {}, "labels": ["monitor", "high_concern"], "alignment_status": "unaligned_legacy"}


def _confidence_bucket(score: float) -> str:
    if score >= 0.9:
        return "very_high"
    if score >= 0.78:
        return "high"
    if score >= 0.64:
        return "medium"
    return "low"


def _prediction_confidence(scores: dict[str, float]) -> float:
    return max(scores.values()) if scores else 0.0


def _payload_signal(modality: str, payload: dict[str, Any]) -> tuple[str, float]:
    text = flatten_payload(payload).lower()
    if modality == "vision":
        if any(token in text for token in ("spiculated", "irregular", "architectural distortion")):
            return "positive", 0.92
        if any(token in text for token in ("circumscribed", "fibroadenoma", "well defined")):
            return "negative", 0.88
        if "asymmetry" in text:
            return "positive", 0.7
    elif modality == "ehr":
        age = payload.get("age") if isinstance(payload, dict) else None
        family_history = payload.get("family_history") if isinstance(payload, dict) else None
        if family_history is True or (isinstance(age, int) and age >= 60):
            return "positive", 0.86
        if family_history is False and isinstance(age, int) and age < 50:
            return "negative", 0.84
    elif modality == "genomics":
        if any(token in text for token in ("pathogenic", "likely_pathogenic", "brca1 pathogenic", "brca2 likely_pathogenic")):
            return "positive", 0.95
        if any(token in text for token in ("wildtype", "benign", "synonymous")):
            return "negative", 0.9
    elif modality == "literature":
        if any(token in text for token in ("aggressive", "triple negative", "surveillance", "brca associated", "brca")):
            return "positive", 0.78
        if any(token in text for token in ("fibroadenoma", "benign")):
            return "negative", 0.76
    return "unknown", 0.5


def _source_tag(case_payload: dict[str, Any], modality: str) -> str:
    sample_id = str(case_payload.get("sample_id", "unknown"))
    return f"{sample_id}::{modality}"


def _build_verifier_features(case_payload: dict[str, Any], predictions: list[Any]) -> tuple[str, dict[str, float]]:
    parts: dict[str, dict[str, Any]] = {}
    positive_modalities = 0
    high_conf_positive = 0
    weighted_positive = 0.0
    weighted_negative = 0.0
    weighted_total = 0.0
    source_mix: dict[str, int] = {}
    payload_positive_modalities = 0
    payload_negative_modalities = 0
    payload_high_conf_positive = 0
    for prediction in predictions:
        predicted_confidence = round(_prediction_confidence(prediction.scores), 4)
        predicted_signal = "positive" if prediction.predicted_label in POSITIVE_LABELS[prediction.modality] else "negative"
        payload = case_payload.get(prediction.modality, {})
        payload_signal, payload_strength = _payload_signal(prediction.modality, payload)
        if payload_signal == "positive":
            signal = "positive"
            confidence_score = round(max(predicted_confidence, payload_strength), 4)
            payload_positive_modalities += 1
            if payload_strength >= 0.78:
                payload_high_conf_positive += 1
        elif payload_signal == "negative":
            signal = "negative"
            confidence_score = round(max(1.0 - predicted_confidence, payload_strength), 4)
            payload_negative_modalities += 1
        else:
            signal = predicted_signal
            confidence_score = predicted_confidence
        source = _source_tag(case_payload, prediction.modality)
        source_mix[source] = source_mix.get(source, 0) + 1
        weighted_total += MODALITY_WEIGHTS[prediction.modality]
        if signal == "positive":
            positive_modalities += 1
            if confidence_score >= 0.78:
                high_conf_positive += 1
            weighted_positive += MODALITY_WEIGHTS[prediction.modality] * (0.45 + confidence_score)
        else:
            weighted_negative += MODALITY_WEIGHTS[prediction.modality] * (0.35 + confidence_score * 0.5)
        parts[prediction.modality] = {
            "presence": "present",
            "sample_id": prediction.sample_id,
            "label": prediction.predicted_label,
            "signal": signal,
            "payload_signal": payload_signal,
            "confidence_bucket": _confidence_bucket(confidence_score),
            "confidence_score": confidence_score,
            "source": source,
            "text": flatten_payload(payload),
        }
    risk_score = round((weighted_positive - weighted_negative) / (weighted_total or 1.0), 4)
    consensus = "mixed" if 0 < positive_modalities < len(predictions) else ("positive" if positive_modalities else "negative")
    feature_payload = {
        "decision_context": "fusion_verifier_inference_row",
        "risk_score_bucket": _confidence_bucket(min(0.98, max(0.08, abs(risk_score)))),
        "positive_modalities": positive_modalities,
        "present_modalities": len(predictions),
        "high_conf_positive": high_conf_positive,
        "payload_positive_modalities": payload_positive_modalities,
        "payload_negative_modalities": payload_negative_modalities,
        "payload_high_conf_positive": payload_high_conf_positive,
        "consensus": consensus,
        "source_mix": source_mix,
    }
    for modality in ("vision", "ehr", "genomics", "literature"):
        feature_payload[modality] = parts.get(modality, {"presence": "missing", "signal": "unknown"})
    heuristic_scores = {
        "high_concern": round(
            max(
                0.0,
                min(
                    1.0,
                    0.12
                    + (0.14 * positive_modalities)
                    + (0.08 * high_conf_positive)
                    + (0.2 * payload_positive_modalities)
                    + (0.08 * payload_high_conf_positive)
                    + (0.22 if parts.get("vision", {}).get("signal") == "positive" and parts.get("ehr", {}).get("signal") == "positive" else 0.0)
                    + (0.18 if parts.get("genomics", {}).get("signal") == "positive" and parts.get("vision", {}).get("signal") == "positive" else 0.0)
                    + (0.18 * max(0.0, risk_score))
                    - (0.14 * payload_negative_modalities),
                ),
            ),
            4,
        )
    }
    heuristic_scores["monitor"] = round(1.0 - heuristic_scores["high_concern"], 4)
    return flatten_payload(feature_payload), heuristic_scores


def _normalize_contributions(weighted_scores: dict[str, float]) -> dict[str, float]:
    total = sum(weighted_scores.values()) or 1.0
    return {key: round(value / total, 4) for key, value in weighted_scores.items()}


def _structured_output(
    case_payload: dict[str, Any],
    predictions: list[Any],
    verifier_scores: dict[str, float],
    verifier: dict[str, Any],
) -> dict[str, Any]:
    risk_score = round(float(verifier_scores.get("high_concern", 0.0)), 4)
    if risk_score <= 0.33:
        risk_classification = "low"
    elif risk_score <= 0.66:
        risk_classification = "intermediate"
    else:
        risk_classification = "high"

    weighted_confidences: dict[str, float] = {}
    modality_predictions: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        confidence = round(_prediction_confidence(prediction.scores), 4)
        weighted_confidences[prediction.modality] = MODALITY_WEIGHTS[prediction.modality] * confidence
        modality_predictions[prediction.modality] = {
            "class": prediction.predicted_label,
            "confidence": confidence,
        }
    modality_contributions = _normalize_contributions(weighted_confidences)
    alignment_status = str(verifier.get("alignment_status", "unaligned_legacy"))
    fused_label = "high_concern" if risk_score >= 0.5 else "monitor"
    return {
        "patient_id": case_payload["sample_id"],
        "risk_classification": risk_classification,
        "risk_score": risk_score,
        "confidence": round(max(verifier_scores.values()) if verifier_scores else 0.0, 4),
        "modality_contributions": modality_contributions,
        "modality_predictions": modality_predictions,
        "alignment_status": alignment_status,
        "sample_id": case_payload["sample_id"],
        "fused_label": fused_label,
        "agent_predictions": [pred.__dict__ for pred in predictions],
        "verifier_scores": verifier_scores,
    }


def run_case(case_payload: dict[str, Any], repo_root: Path | None = None) -> dict[str, Any]:
    sample_id = case_payload["sample_id"]
    root = repo_root or Path(__file__).resolve().parents[1]
    agents = [
        VisionAgent(_artifact_path(root, "vision")),
        EHRAgent(_artifact_path(root, "ehr")),
        GenomicsAgent(_artifact_path(root, "genomics")),
        LiteratureAgent(_artifact_path(root, "literature")),
    ]
    predictions = [agent.predict(sample_id, case_payload.get(agent.modality, {})) for agent in agents]
    verifier = _load_verifier(root)
    verifier_features, heuristic_scores = _build_verifier_features(case_payload, predictions)
    verifier_scores = score_with_prototypes(verifier_features, verifier.get("prototypes", {}))
    if verifier_scores:
        # Blend learned prototype scores with a rule-based prior so strong malignant + high-risk cases are not suppressed.
        verifier_scores = {
            label: round((0.65 * verifier_scores.get(label, 0.0)) + (0.35 * heuristic_scores.get(label, 0.0)), 4)
            for label in ("monitor", "high_concern")
        }
    else:
        verifier_scores = heuristic_scores
    return _structured_output(case_payload, predictions, verifier_scores, verifier)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal fusion")
    parser.add_argument("--case-path", default=None)
    parser.add_argument("--all-sample-cases", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    repo_root = settings.repo_root
    case_paths = [Path(args.case_path)] if args.case_path else sorted((repo_root / "sample_cases").glob("*.json"))
    if not args.all_sample_cases:
        case_paths = case_paths[:1]

    results = [run_case(read_json(case_path), repo_root=repo_root) for case_path in case_paths]
    outputs_root = repo_root / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    (outputs_root / "sample_case_results.json").write_text(json.dumps(results, indent=2))
    # Preserve the legacy filename for downstream compatibility.
    (outputs_root / "fused_predictions.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
