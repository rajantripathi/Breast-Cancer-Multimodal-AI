from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.statistics import (
    binary_brier_score,
    bootstrap_confidence_interval,
    calibration_slope_intercept,
    decision_curve,
    expected_calibration_error,
    harrell_c_index,
)
from evaluation.subgroups import summarize_survival_subgroups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize external Stage 2 validation results")
    parser.add_argument("--internal-root", type=Path, required=True)
    parser.add_argument("--metabric-cg-root", type=Path, required=True)
    parser.add_argument("--metabric-g-root", type=Path, required=True)
    parser.add_argument("--metabric-clinical", type=Path, required=True)
    parser.add_argument("--cptac-alignment-probe", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def seed_sort_key(path: Path) -> str:
    return path.name


def load_predictions(seed_dir: Path) -> list[dict[str, Any]]:
    predictions_path = seed_dir / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions.json in {seed_dir}")
    return json.loads(predictions_path.read_text())


def load_seed_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / "summary.json").exists()],
        key=seed_sort_key,
    )


def per_seed_c_indices(seed_dirs: list[Path]) -> dict[str, float]:
    values: dict[str, float] = {}
    for seed_dir in seed_dirs:
        summary = json.loads((seed_dir / "summary.json").read_text())
        values[seed_dir.name] = float(summary.get("c_index_mean", summary.get("c_index", 0.0)))
    return values


def aggregate_patient_predictions(seed_dirs: list[Path]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed_dir in seed_dirs:
        for row in load_predictions(seed_dir):
            grouped[str(row["sample_id"])].append(row)

    aggregated: list[dict[str, Any]] = []
    for sample_id, rows in sorted(grouped.items()):
        first = rows[0]
        mean_risk = float(np.mean([float(row["risk_score"]) for row in rows]))
        aggregated.append(
            {
                "sample_id": sample_id,
                "true_label": first["true_label"],
                "risk_score": mean_risk,
                "survival_time": float(first["survival_time"]),
                "event_observed": int(first["event_observed"]),
            }
        )
    return aggregated


def cindex_ci(predictions: list[dict[str, Any]], iterations: int, seed: int) -> tuple[float, float]:
    return bootstrap_confidence_interval(
        harrell_c_index,
        [float(row["survival_time"]) for row in predictions],
        [float(row["risk_score"]) for row in predictions],
        [int(row["event_observed"]) for row in predictions],
        iterations=iterations,
        seed=seed,
        stratify=False,
    )


def independent_cindex_delta(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    if not baseline or not candidate:
        return {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    deltas: list[float] = []
    base_n = len(baseline)
    cand_n = len(candidate)
    for _ in range(iterations):
        base_idx = rng.randint(0, base_n, size=base_n)
        cand_idx = rng.randint(0, cand_n, size=cand_n)
        base_rows = [baseline[index] for index in base_idx]
        cand_rows = [candidate[index] for index in cand_idx]
        base_metric = harrell_c_index(
            [float(row["survival_time"]) for row in base_rows],
            [float(row["risk_score"]) for row in base_rows],
            [int(row["event_observed"]) for row in base_rows],
        )
        cand_metric = harrell_c_index(
            [float(row["survival_time"]) for row in cand_rows],
            [float(row["risk_score"]) for row in cand_rows],
            [int(row["event_observed"]) for row in cand_rows],
        )
        deltas.append(float(cand_metric - base_metric))
    deltas.sort()
    return {
        "mean_delta": round(float(np.mean(deltas)), 4),
        "ci_lower": round(float(deltas[int(0.025 * (len(deltas) - 1))]), 4),
        "ci_upper": round(float(deltas[int(0.975 * (len(deltas) - 1))]), 4),
    }


def prediction_calibration(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [1 if row["true_label"] == "high_concern" else 0 for row in predictions]
    scores = [float(row["risk_score"]) for row in predictions]
    hard_predictions = [1 if score >= 0.5 else 0 for score in scores]
    confidences = [score if pred == 1 else 1.0 - score for score, pred in zip(scores, hard_predictions)]
    return {
        "brier_score": round(float(binary_brier_score(labels, scores)), 6),
        "ece": round(float(expected_calibration_error(labels, confidences, hard_predictions)), 6),
        "calibration_slope_intercept": calibration_slope_intercept(labels, scores),
        "decision_curve": decision_curve(labels, scores),
    }


def age_tertile_summary(predictions: list[dict[str, Any]], clinical_lookup: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ages = [
        float(values["age_at_diagnosis"])
        for values in clinical_lookup.values()
        if values.get("age_at_diagnosis") not in ("", None)
    ]
    if not ages:
        return {}
    low_cut, high_cut = np.quantile(np.asarray(ages, dtype=np.float64), [1 / 3, 2 / 3]).tolist()

    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        values = clinical_lookup.get(str(row["sample_id"]), {})
        age_value = values.get("age_at_diagnosis")
        if age_value in ("", None):
            group = "Unknown"
        else:
            age_days = float(age_value)
            if age_days <= low_cut:
                group = "Age tertile 1"
            elif age_days <= high_cut:
                group = "Age tertile 2"
            else:
                group = "Age tertile 3"
        rows_by_group[group].append(row)
    return summarize_survival_subgroups(
        [
            dict(row, age_group=group)
            for group, rows in rows_by_group.items()
            for row in rows
        ],
        "age_group",
        horizon_days=1825.0,
        min_group_size=20,
        min_events=5,
    )


def metabric_clinical_lookup(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path).copy()
    lookup: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        lookup[str(row["patient_barcode"])] = row
    return lookup


def attach_subgroups(predictions: list[dict[str, Any]], clinical_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for row in predictions:
        patient = str(row["sample_id"])
        clinical = clinical_lookup.get(patient, {})
        merged = dict(row)
        merged["tumor_stage"] = clinical.get("tumor_stage", "UNKNOWN")
        merged["pathologic_stage"] = clinical.get("pathologic_stage", "UNKNOWN")
        merged["er_status_by_ihc"] = clinical.get("er_status_by_ihc", "UNKNOWN")
        attached.append(merged)
    return attached


def summarize_config(
    root: Path,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seed_dirs = load_seed_dirs(root)
    seed_cindex = per_seed_c_indices(seed_dirs)
    predictions = aggregate_patient_predictions(seed_dirs)
    c_index = harrell_c_index(
        [float(row["survival_time"]) for row in predictions],
        [float(row["risk_score"]) for row in predictions],
        [int(row["event_observed"]) for row in predictions],
    )
    ci_low, ci_high = cindex_ci(predictions, bootstrap_iterations, bootstrap_seed)
    return {
        "seed_dirs": [path.name for path in seed_dirs],
        "seed_mean_c_index": {seed: round(value, 4) for seed, value in seed_cindex.items()},
        "mean_c_index_across_seeds": round(float(np.mean(list(seed_cindex.values()))), 4),
        "ensemble_patient_level_c_index": round(float(c_index), 4),
        "bootstrap_95ci": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "n_patients": len(predictions),
        "predictions": predictions,
        "calibration": prediction_calibration(predictions),
    }


def main() -> None:
    args = parse_args()
    internal = summarize_config(
        args.internal_root,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.seed),
    )
    metabric_cg = summarize_config(
        args.metabric_cg_root,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.seed) + 1,
    )
    metabric_g = summarize_config(
        args.metabric_g_root,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.seed) + 2,
    )

    clinical_lookup = metabric_clinical_lookup(args.metabric_clinical)
    metabric_cg_attached = attach_subgroups(metabric_cg["predictions"], clinical_lookup)
    metabric_g_attached = attach_subgroups(metabric_g["predictions"], clinical_lookup)

    metabric_subgroups = {
        "clinical_genomics": {
            "er_status": summarize_survival_subgroups(
                [dict(row, er_status=row.get("er_status_by_ihc", "UNKNOWN")) for row in metabric_cg_attached],
                "er_status",
                horizon_days=1825.0,
            ),
            "stage": summarize_survival_subgroups(
                [dict(row, tumor_stage=row.get("tumor_stage", "UNKNOWN")) for row in metabric_cg_attached],
                "tumor_stage",
                horizon_days=1825.0,
            ),
            "age_tertile": age_tertile_summary(metabric_cg["predictions"], clinical_lookup),
        },
        "genomics_only": {
            "er_status": summarize_survival_subgroups(
                [dict(row, er_status=row.get("er_status_by_ihc", "UNKNOWN")) for row in metabric_g_attached],
                "er_status",
                horizon_days=1825.0,
            ),
            "stage": summarize_survival_subgroups(
                [dict(row, tumor_stage=row.get("tumor_stage", "UNKNOWN")) for row in metabric_g_attached],
                "tumor_stage",
                horizon_days=1825.0,
            ),
            "age_tertile": age_tertile_summary(metabric_g["predictions"], clinical_lookup),
        },
    }

    cptac_status: dict[str, Any] | None = None
    if args.cptac_alignment_probe and args.cptac_alignment_probe.exists():
        cptac_status = json.loads(args.cptac_alignment_probe.read_text())

    external_summary = {
        "internal_tcga_simple_fusion_mean": {
            key: value for key, value in internal.items() if key != "predictions"
        },
        "metabric": {
            "clinical_genomics": {
                key: value for key, value in metabric_cg.items() if key != "predictions"
            },
            "genomics_only": {
                key: value for key, value in metabric_g.items() if key != "predictions"
            },
            "subgroups": metabric_subgroups,
            "endpoint_note": "METABRIC uses overall survival as a proxy for the internal TCGA PFI endpoint; 5-year OS label is used for binary calibration while c-index uses OS time/event.",
            "assay_note": "METABRIC uses microarray expression; pathway features mirror the repository's TCGA mean-over-hallmark-gene construction rather than full GSVA.",
        },
        "comparisons": {
            "metabric_cg_vs_internal_cindex_delta": independent_cindex_delta(
                internal["predictions"],
                metabric_cg["predictions"],
                iterations=int(args.bootstrap_iterations),
                seed=int(args.seed) + 3,
            ),
            "metabric_g_vs_internal_cindex_delta": independent_cindex_delta(
                internal["predictions"],
                metabric_g["predictions"],
                iterations=int(args.bootstrap_iterations),
                seed=int(args.seed) + 4,
            ),
            "comparison_note": "Internal vs external c-index deltas use independent bootstrap because these are different cohorts; DeLong is not applicable to cross-cohort survival c-index.",
        },
        "cptac_brca": cptac_status
        or {
            "status": "outcome_unavailable_public",
            "note": "CPTAC image/RNA alignment is available, but a public survival endpoint was not identified in this branch.",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(external_summary, indent=2) + "\n")

    md_lines = [
        "# External Validation",
        "",
        "## METABRIC",
        "",
        f"- Internal TCGA simple fusion mean C-index: `{internal['mean_c_index_across_seeds']:.4f}` "
        f"(ensemble patient-level `{internal['ensemble_patient_level_c_index']:.4f}`, 95% CI "
        f"`[{internal['bootstrap_95ci'][0]:.4f}, {internal['bootstrap_95ci'][1]:.4f}]`).",
        f"- METABRIC clinical+genomics C-index: `{metabric_cg['mean_c_index_across_seeds']:.4f}` "
        f"(ensemble patient-level `{metabric_cg['ensemble_patient_level_c_index']:.4f}`, 95% CI "
        f"`[{metabric_cg['bootstrap_95ci'][0]:.4f}, {metabric_cg['bootstrap_95ci'][1]:.4f}]`).",
        f"- METABRIC genomics-only C-index: `{metabric_g['mean_c_index_across_seeds']:.4f}` "
        f"(ensemble patient-level `{metabric_g['ensemble_patient_level_c_index']:.4f}`, 95% CI "
        f"`[{metabric_g['bootstrap_95ci'][0]:.4f}, {metabric_g['bootstrap_95ci'][1]:.4f}]`).",
        f"- Internal-to-external delta (clinical+genomics): "
        f"`{external_summary['comparisons']['metabric_cg_vs_internal_cindex_delta']['mean_delta']:.4f}` "
        f"with bootstrap CI "
        f"`[{external_summary['comparisons']['metabric_cg_vs_internal_cindex_delta']['ci_lower']:.4f}, "
        f"{external_summary['comparisons']['metabric_cg_vs_internal_cindex_delta']['ci_upper']:.4f}]`.",
        f"- Internal-to-external delta (genomics-only): "
        f"`{external_summary['comparisons']['metabric_g_vs_internal_cindex_delta']['mean_delta']:.4f}` "
        f"with bootstrap CI "
        f"`[{external_summary['comparisons']['metabric_g_vs_internal_cindex_delta']['ci_lower']:.4f}, "
        f"{external_summary['comparisons']['metabric_g_vs_internal_cindex_delta']['ci_upper']:.4f}]`.",
        "",
        "## CPTAC-BRCA",
        "",
        f"- Status: `{external_summary['cptac_brca'].get('status', 'unknown')}`.",
        f"- Note: {external_summary['cptac_brca'].get('note', 'Public survival outcomes were not available for external c-index evaluation.')}",
        "",
        "## Interpretation",
        "",
        "- METABRIC is a non-vision external cohort with endpoint and assay mismatch relative to TCGA PFI/RNA-seq; the quantitative result should be framed as external robustness rather than a like-for-like clinical replication.",
        "- CPTAC-BRCA remains image/RNA alignment-ready but not yet quantitatively evaluable for survival in the public branch because a usable public outcome source was not identified.",
    ]
    args.output_markdown.write_text("\n".join(md_lines) + "\n")
    print(f"saved {args.output_json}")
    print(f"saved {args.output_markdown}")


if __name__ == "__main__":
    main()
