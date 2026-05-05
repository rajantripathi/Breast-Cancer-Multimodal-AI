from __future__ import annotations

"""Utilities for subgroup analysis on TCGA-style survival artifacts."""

from pathlib import Path
from typing import Any

import pandas as pd

from .statistics import binary_auroc, harrell_c_index, survival_binary_labels_at_horizon

TCGA_SUBGROUP_COLUMNS = {
    "tumor_stage": "tumor_stage",
    "pathologic_stage": "pathologic_stage",
    "er_status": "er_status_by_ihc",
    "pr_status": "pr_status_by_ihc",
    "her2_status": "her2_status_by_ihc",
}


def resolve_optional_repo_path(path_hint: str | Path | None, repo_root: str | Path) -> Path | None:
    if not path_hint:
        return None
    candidate = Path(path_hint)
    if candidate.exists():
        return candidate
    repo_candidate = Path(repo_root) / candidate
    if repo_candidate.exists():
        return repo_candidate
    return None


def _normalize_stage(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return "UNKNOWN"
    if "IV" in text:
        return "Stage IV"
    if "III" in text:
        return "Stage III"
    if "II" in text:
        return "Stage II"
    if "I" in text:
        return "Stage I"
    return text


def _normalize_receptor(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return "UNKNOWN"
    if "POS" in text:
        return "Positive"
    if "NEG" in text:
        return "Negative"
    if "EQUIV" in text:
        return "Equivocal"
    return text


def load_tcga_clinical_subgroups(clinical_csv: str | Path) -> dict[str, dict[str, str]]:
    frame = pd.read_csv(clinical_csv).copy()
    barcode_column = "bcr_patient_barcode" if "bcr_patient_barcode" in frame.columns else "patient_barcode"
    frame["patient_barcode"] = frame[barcode_column].astype(str).str.upper().str.slice(0, 12)
    lookup: dict[str, dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        barcode = str(row.get("patient_barcode", "")).upper()[:12]
        if not barcode:
            continue
        lookup[barcode] = {
            "tumor_stage": _normalize_stage(row.get("tumor_stage")),
            "pathologic_stage": _normalize_stage(row.get("pathologic_stage")),
            "er_status": _normalize_receptor(row.get("er_status_by_ihc")),
            "pr_status": _normalize_receptor(row.get("pr_status_by_ihc")),
            "her2_status": _normalize_receptor(row.get("her2_status_by_ihc")),
        }
    return lookup


def attach_tcga_subgroups(predictions: list[dict[str, Any]], clinical_lookup: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for row in predictions:
        barcode = str(row.get("sample_id", row.get("study_id", ""))).upper()[:12]
        subgroup_values = clinical_lookup.get(barcode, {})
        merged = dict(row)
        merged.update(subgroup_values)
        attached.append(merged)
    return attached


def summarize_survival_subgroups(
    predictions: list[dict[str, Any]],
    group_key: str,
    *,
    horizon_days: float,
    min_group_size: int = 20,
    min_events: int = 5,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        group_value = str(row.get(group_key, "UNKNOWN") or "UNKNOWN")
        groups.setdefault(group_value, []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for group_value, rows in sorted(groups.items()):
        survival_times = [float(row["survival_time"]) for row in rows]
        event_observed = [int(row["event_observed"]) for row in rows]
        risk_scores = [float(row["risk_score"]) for row in rows]
        n_events = int(sum(event_observed))
        horizon_payload = survival_binary_labels_at_horizon(survival_times, event_observed, risk_scores, horizon_days)
        labels = horizon_payload["labels"]
        scores = horizon_payload["scores"]
        c_index = harrell_c_index(survival_times, risk_scores, event_observed) if len(rows) >= min_group_size and n_events >= min_events else None
        auroc = binary_auroc(labels, scores) if len(scores) >= min_group_size and sum(labels) >= min_events and len(set(labels)) > 1 else None
        summary[group_value] = {
            "n_patients": len(rows),
            "n_events": n_events,
            "event_rate": round(n_events / len(rows), 4) if rows else 0.0,
            "eligible_auroc_n": int(horizon_payload["n_eligible"]),
            "eligible_auroc_events": int(horizon_payload["n_events"]),
            "c_index": round(float(c_index), 4) if c_index is not None else None,
            "time_dependent_auroc": round(float(auroc), 4) if auroc is not None else None,
        }
    return summary
