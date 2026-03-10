from __future__ import annotations

from collections import Counter
from itertools import cycle
from pathlib import Path

from config import load_settings
from data.common import flatten_payload, read_jsonl, stable_shuffle, write_jsonl


MODALITIES = ("vision", "ehr", "genomics", "literature")
MODALITY_WEIGHTS = {"vision": 1.2, "ehr": 1.0, "genomics": 1.35, "literature": 0.55}
POSITIVE_LABELS = {
    "vision": {"malignant"},
    "ehr": {"high_risk"},
    "genomics": {"pathogenic_variant"},
    "literature": {"supportive_evidence"},
}
MISSING_PATTERNS = [(), ("literature",), ("ehr",), ("vision",), ("genomics",), ("literature", "ehr")]
SCENARIOS = [
    {"vision": "positive", "ehr": "positive", "genomics": "positive", "literature": "positive"},
    {"vision": "negative", "ehr": "negative", "genomics": "negative", "literature": "negative"},
    {"vision": "negative", "ehr": "positive", "genomics": "positive", "literature": "positive"},
    {"vision": "positive", "ehr": "negative", "genomics": "negative", "literature": "positive"},
    {"vision": "negative", "ehr": "negative", "genomics": "positive", "literature": "negative"},
    {"vision": "positive", "ehr": "positive", "genomics": "negative", "literature": "negative"},
]


def _partition_rows(rows: list[dict], modality: str) -> tuple[list[dict], list[dict]]:
    positive = [row for row in rows if row["label"] in POSITIVE_LABELS[modality]]
    negative = [row for row in rows if row["label"] not in POSITIVE_LABELS[modality]]
    return stable_shuffle(positive, seed=len(rows) + len(modality)), stable_shuffle(negative, seed=2 * len(rows) + len(modality))


def _source_tag(row: dict) -> str:
    metadata = row.get("metadata", {})
    source = metadata.get("source", "unknown")
    source_tag = Path(str(source)).stem or "unknown"
    suffixes = [str(metadata.get(key, "")).strip().replace(" ", "_") for key in ("variant", "query") if metadata.get(key)]
    return "::".join([source_tag] + suffixes) if suffixes else source_tag


def _confidence_score(row: dict, modality: str) -> float:
    text = str(row.get("text", "")).lower()
    label = str(row.get("label", ""))
    score = 0.52 + (0.14 if label in POSITIVE_LABELS[modality] else 0.05)
    if modality == "vision":
        if any(token in text for token in ("spiculated", "irregular", "architectural")):
            score += 0.2
        if any(token in text for token in ("circumscribed", "well defined", "rounded")):
            score += 0.1
    elif modality == "ehr":
        if any(token in text for token in ("family_history true", "lymph_node yes", "grade 3", "age 6", "age 7")):
            score += 0.16
        if any(token in text for token in ("family_history false", "tumor_size 1", "nuclei 1")):
            score += 0.08
    elif modality == "genomics":
        if any(token in text for token in ("pathogenic", "del", "frameshift", "brca1", "brca2")):
            score += 0.18
        if "wildtype" in text or "synonymous" in text:
            score += 0.08
    elif modality == "literature":
        if any(token in text for token in ("screening", "surveillance", "trial", "meta-analysis", "guideline")):
            score += 0.12
        if "variant" in text or "brca" in text:
            score += 0.08
    return round(min(0.98, max(0.08, score)), 4)


def _confidence_bucket(score: float) -> str:
    if score >= 0.9:
        return "very_high"
    if score >= 0.78:
        return "high"
    if score >= 0.64:
        return "medium"
    return "low"


def _enrich_row(row: dict, modality: str) -> dict:
    confidence_score = _confidence_score(row, modality)
    signal = "positive" if row["label"] in POSITIVE_LABELS[modality] else "negative"
    return {
        **row,
        "signal": signal,
        "confidence_score": confidence_score,
        "confidence_bucket": _confidence_bucket(confidence_score),
        "source_tag": _source_tag(row),
    }


def _risk_summary(parts: dict[str, dict]) -> dict[str, object]:
    present = [part for part in parts.values() if not part.get("missing")]
    weighted_total = sum(MODALITY_WEIGHTS[part["modality"]] for part in present) or 1.0
    weighted_positive = sum(
        MODALITY_WEIGHTS[part["modality"]] * (0.45 + part["confidence_score"])
        for part in present
        if part["signal"] == "positive"
    )
    weighted_negative = sum(
        MODALITY_WEIGHTS[part["modality"]] * (0.35 + part["confidence_score"] * 0.5)
        for part in present
        if part["signal"] == "negative"
    )
    positive_modalities = sum(part["signal"] == "positive" for part in present)
    high_conf_positive = sum(part["signal"] == "positive" and part["confidence_score"] >= 0.78 for part in present)
    genomics_positive = parts["genomics"]["signal"] == "positive" and not parts["genomics"].get("missing")
    vision_positive = parts["vision"]["signal"] == "positive" and not parts["vision"].get("missing")
    score = (weighted_positive - weighted_negative) / weighted_total
    high_concern = score >= 0.2 or high_conf_positive >= 2 or (genomics_positive and vision_positive) or (positive_modalities >= 3)
    consensus = "mixed" if 0 < positive_modalities < len(present) else ("positive" if positive_modalities else "negative")
    return {
        "target_label": "high_concern" if high_concern else "monitor",
        "risk_score": round(score, 4),
        "positive_modalities": positive_modalities,
        "present_modalities": len(present),
        "consensus": consensus,
        "source_mix": dict(Counter(part["source_tag"] for part in present)),
    }


def _bundle_text(parts: dict[str, dict], summary: dict[str, object]) -> str:
    payload = {
        "target_label": summary["target_label"],
        "risk_score_bucket": _confidence_bucket(min(0.98, max(0.08, abs(float(summary["risk_score"]))))),
        "positive_modalities": summary["positive_modalities"],
        "present_modalities": summary["present_modalities"],
        "consensus": summary["consensus"],
        "source_mix": summary["source_mix"],
    }
    for modality in MODALITIES:
        part = parts[modality]
        if part.get("missing"):
            payload[modality] = {"presence": "missing", "signal": "unknown"}
            continue
        payload[modality] = {
            "presence": "present",
            "sample_id": part["sample_id"],
            "label": part["label"],
            "signal": part["signal"],
            "confidence_bucket": part["confidence_bucket"],
            "confidence_score": part["confidence_score"],
            "source": part["source_tag"],
            "text": part["text"],
        }
    return flatten_payload(payload)


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "verifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    modality_rows = {
        modality: [_enrich_row(row, modality) for row in read_jsonl(settings.processed_data_root / modality / "dataset.jsonl")]
        for modality in MODALITIES
    }
    positive, negative = {}, {}
    for modality, rows in modality_rows.items():
        positive[modality], negative[modality] = _partition_rows(rows, modality)

    target_size = min(
        12000,
        max(
            2000,
            4 * min(
                len(positive["vision"]) + len(negative["vision"]),
                len(positive["ehr"]) + len(negative["ehr"]),
                len(positive["genomics"]) + len(negative["genomics"]),
                len(positive["literature"]) + len(negative["literature"]),
            ),
        ),
    )

    rows: list[dict] = []
    positive_cycles = {modality: cycle(positive[modality] or negative[modality]) for modality in modality_rows}
    negative_cycles = {modality: cycle(negative[modality] or positive[modality]) for modality in modality_rows}

    for index in range(target_size):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        missing = set(MISSING_PATTERNS[index % len(MISSING_PATTERNS)])
        parts: dict[str, dict] = {}
        for modality in MODALITIES:
            if modality in missing:
                parts[modality] = {
                    "modality": modality,
                    "missing": True,
                    "signal": "unknown",
                    "confidence_score": 0.0,
                    "confidence_bucket": "missing",
                    "source_tag": "missing",
                }
                continue
            desired_signal = scenario[modality]
            base_row = next(positive_cycles[modality]) if desired_signal == "positive" else next(negative_cycles[modality])
            parts[modality] = {**base_row, "modality": modality, "missing": False}

        summary = _risk_summary(parts)
        rows.append(
            {
                "sample_id": f"verifier_bundle_{index:05d}",
                "label": summary["target_label"],
                "text": _bundle_text(parts, summary),
                "metadata": {
                    "risk_score": summary["risk_score"],
                    "consensus": summary["consensus"],
                    "positive_modalities": summary["positive_modalities"],
                    "present_modalities": summary["present_modalities"],
                    "missing_modalities": sorted(missing),
                    "vision_id": parts["vision"].get("sample_id", ""),
                    "ehr_id": parts["ehr"].get("sample_id", ""),
                    "genomics_id": parts["genomics"].get("sample_id", ""),
                    "literature_id": parts["literature"].get("sample_id", ""),
                    "source_mix": summary["source_mix"],
                },
            }
        )

    write_jsonl(out_dir / "dataset.jsonl", rows)
    print(f"verifier processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
