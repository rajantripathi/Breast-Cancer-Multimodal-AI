from __future__ import annotations

"""Build METABRIC external-validation assets from public cBioPortal data.

This downloader intentionally mirrors the existing TCGA pathway pipeline in
this repository. The trained Stage 2 checkpoints were fit on "Hallmark pathway"
features produced by averaging expression across genes in each Hallmark set,
not by a full GSVA implementation. To avoid silently changing the feature
distribution at external validation time, this script reuses that same
mean-over-genes construction for METABRIC.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
import urllib.request

import pandas as pd
import torch

from data.common import write_json
from data.preprocess.preprocess_tcga_genomics import _load_hallmark_pathways


CBIO_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "brca_metabric"
DEFAULT_MOLECULAR_PROFILE = "brca_metabric_mrna"
DEFAULT_SAMPLE_LIST = "brca_metabric_all"
DEFAULT_ATTRIBUTES = [
    "AGE_AT_DIAGNOSIS",
    "ER_IHC",
    "ER_STATUS",
    "HER2_SNP6",
    "HER2_STATUS",
    "OS_MONTHS",
    "OS_STATUS",
    "PR_STATUS",
    "TUMOR_STAGE",
    "GRADE",
    "VITAL_STATUS",
    "HISTOLOGICAL_SUBTYPE",
]
POSITIVE_VITAL_STATUS = {"DECEASED", "DEAD", "1:DECEASED", "DECEASED/PROGRESSION"}
NEGATIVE_VITAL_STATUS = {"LIVING", "ALIVE", "0:LIVING"}
DEFAULT_HORIZON_DAYS = 1825.0


def _request_json(path: str, payload: Any | None = None) -> Any:
    url = f"{CBIO_BASE}{path}"
    if payload is None:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fetch_patients(page_size: int) -> list[str]:
    patient_ids: list[str] = []
    page = 0
    while True:
        batch = _request_json(
            f"/studies/{STUDY_ID}/patients?projection=SUMMARY&pageSize={int(page_size)}&pageNumber={page}"
        )
        if not batch:
            break
        patient_ids.extend(str(item["patientId"]) for item in batch if item.get("patientId"))
        if len(batch) < int(page_size):
            break
        page += 1
    return patient_ids


def _fetch_patient_clinical(patient_ids: list[str], attribute_ids: list[str], batch_size: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {patient_id: {"patient_id": patient_id} for patient_id in patient_ids}
    for start in range(0, len(patient_ids), batch_size):
        chunk = patient_ids[start : start + batch_size]
        payload = {"ids": chunk, "attributeIds": attribute_ids}
        rows = _request_json(
            f"/studies/{STUDY_ID}/clinical-data/fetch?clinicalDataType=PATIENT&projection=SUMMARY",
            payload,
        )
        for row in rows:
            patient_id = str(row.get("patientId") or "").strip()
            attribute_id = str(row.get("clinicalAttributeId") or "").strip()
            if not patient_id or not attribute_id:
                continue
            grouped.setdefault(patient_id, {"patient_id": patient_id})[attribute_id] = row.get("value")
        print(
            f"downloaded clinical data for {min(start + batch_size, len(patient_ids))}/{len(patient_ids)} patients",
            flush=True,
        )
    return [grouped[patient_id] for patient_id in patient_ids]


def _fetch_gene_map(hugo_symbols: list[str], batch_size: int) -> tuple[dict[str, int], list[str]]:
    gene_map: dict[str, int] = {}
    missing: list[str] = []
    for start in range(0, len(hugo_symbols), batch_size):
        chunk = hugo_symbols[start : start + batch_size]
        rows = _request_json("/genes/fetch?geneIdType=HUGO_GENE_SYMBOL&projection=SUMMARY", chunk)
        found = {str(row.get("hugoGeneSymbol", "")).upper(): int(row["entrezGeneId"]) for row in rows if row.get("entrezGeneId")}
        for symbol in chunk:
            normalized = symbol.upper()
            if normalized in found:
                gene_map[normalized] = found[normalized]
            else:
                missing.append(normalized)
        print(f"mapped {min(start + batch_size, len(hugo_symbols))}/{len(hugo_symbols)} hallmark genes", flush=True)
    return gene_map, sorted(set(missing))


def _build_expression_matrix(
    molecular_profile_id: str,
    sample_list_id: str,
    entrez_gene_ids: list[int],
    batch_size: int,
    entrez_to_symbol: dict[int, str],
) -> pd.DataFrame:
    matrix_rows: dict[str, dict[str, float]] = {}
    for start in range(0, len(entrez_gene_ids), batch_size):
        chunk = entrez_gene_ids[start : start + batch_size]
        payload = {"sampleListId": sample_list_id, "entrezGeneIds": chunk}
        batch = _request_json(
            f"/molecular-profiles/{molecular_profile_id}/molecular-data/fetch?projection=SUMMARY",
            payload,
        )
        for row in batch:
            patient_id = str(row.get("patientId") or row.get("sampleId") or "").strip()
            entrez = row.get("entrezGeneId")
            if not patient_id or entrez is None:
                continue
            gene_symbol = entrez_to_symbol.get(int(entrez))
            if not gene_symbol:
                continue
            matrix_rows.setdefault(patient_id, {})[gene_symbol] = float(row.get("value", 0.0))
        print(
            f"downloaded expression for {min(start + batch_size, len(entrez_gene_ids))}/{len(entrez_gene_ids)} genes "
            f"({len(batch)} rows in last batch)",
            flush=True,
        )
    frame = pd.DataFrame.from_dict(matrix_rows, orient="index").fillna(0.0)
    frame.index.name = "patient_barcode"
    return frame.sort_index()


def _normalize_status(value: Any) -> tuple[str, int | None]:
    text = str(value or "").strip().upper()
    if text in POSITIVE_VITAL_STATUS:
        return "Dead", 1
    if text in NEGATIVE_VITAL_STATUS:
        return "Alive", 0
    if "DECEASED" in text or "DEAD" in text:
        return "Dead", 1
    if "LIVING" in text or "ALIVE" in text:
        return "Alive", 0
    return "Unknown", None


def _months_to_days(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(number * 30.4375, 0.0)


def _age_years_to_days(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(number * 365.25, 0.0)


def _binary_label_at_horizon(event_observed: int | None, survival_time_days: float | None, horizon_days: float) -> int | None:
    if event_observed is None or survival_time_days is None:
        return None
    if event_observed == 1:
        return 1 if survival_time_days <= horizon_days else 0
    return 0 if survival_time_days >= horizon_days else None


def _clinical_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    status_text, event_observed = _normalize_status(raw_row.get("OS_STATUS") or raw_row.get("VITAL_STATUS"))
    survival_time_days = _months_to_days(raw_row.get("OS_MONTHS"))
    days_to_death = survival_time_days if event_observed == 1 else ""
    days_to_last_followup = survival_time_days if event_observed == 0 else ""
    gender = str(raw_row.get("SEX") or raw_row.get("GENDER") or "UNKNOWN").strip() or "UNKNOWN"
    return {
        "patient_barcode": str(raw_row["patient_id"]).strip(),
        "age_at_diagnosis": _age_years_to_days(raw_row.get("AGE_AT_DIAGNOSIS")) or "",
        "gender": gender,
        "vital_status": status_text,
        "days_to_death": days_to_death,
        "days_to_last_followup": days_to_last_followup,
        "tumor_stage": str(raw_row.get("TUMOR_STAGE") or "").strip(),
        "pathologic_stage": str(raw_row.get("TUMOR_STAGE") or "").strip(),
        "er_status_by_ihc": str(raw_row.get("ER_IHC") or raw_row.get("ER_STATUS") or "").strip(),
        "pr_status_by_ihc": str(raw_row.get("PR_STATUS") or "").strip(),
        "her2_status_by_ihc": str(raw_row.get("HER2_SNP6") or raw_row.get("HER2_STATUS") or "").strip(),
        "histological_type": str(raw_row.get("HISTOLOGICAL_SUBTYPE") or "").strip(),
    }


def _build_pathway_matrix(matrix: pd.DataFrame, pathways: dict[str, list[str]]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for patient_barcode, patient_row in matrix.iterrows():
        patient_scores: dict[str, float] = {}
        for pathway_name, genes in pathways.items():
            available = [gene for gene in genes if gene in matrix.columns]
            if not available:
                patient_scores[pathway_name] = 0.0
                continue
            patient_scores[pathway_name] = float(patient_row[available].mean())
        rows[str(patient_barcode)] = patient_scores
    pathway_frame = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    pathway_frame.index.name = "patient_barcode"
    return pathway_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare METABRIC external-validation assets from cBioPortal")
    parser.add_argument("--output-dir", required=True, help="Root external/metabric directory")
    parser.add_argument("--crosswalk-output", required=True, help="Path to write data/external METABRIC crosswalk CSV")
    parser.add_argument("--hallmark-gmt", required=True, help="Hallmark GMT used for the TCGA pathway tensors")
    parser.add_argument("--molecular-profile-id", default=DEFAULT_MOLECULAR_PROFILE)
    parser.add_argument("--sample-list-id", default=DEFAULT_SAMPLE_LIST)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--clinical-batch-size", type=int, default=250)
    parser.add_argument("--gene-map-batch-size", type=int, default=500)
    parser.add_argument("--expression-batch-size", type=int, default=32)
    parser.add_argument("--horizon-days", type=float, default=DEFAULT_HORIZON_DAYS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "metadata"
    preprocessed_dir = output_dir / "preprocessed"
    genomics_dir = preprocessed_dir / "genomics_pathways"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    genomics_dir.mkdir(parents=True, exist_ok=True)

    study = _request_json(f"/studies/{STUDY_ID}")
    attributes = _request_json(f"/studies/{STUDY_ID}/clinical-attributes?projection=SUMMARY")
    patient_ids = _fetch_patients(int(args.page_size))
    raw_clinical_rows = _fetch_patient_clinical(
        patient_ids,
        list(DEFAULT_ATTRIBUTES),
        int(args.clinical_batch_size),
    )

    _write_csv(metadata_dir / "clinical_raw.csv", raw_clinical_rows)
    write_json(metadata_dir / "study.json", study)
    write_json(
        metadata_dir / "clinical_attributes.json",
        {
            str(item.get("clinicalAttributeId")): {
                "displayName": item.get("displayName"),
                "patientAttribute": bool(item.get("patientAttribute")),
                "datatype": item.get("datatype"),
            }
            for item in attributes
            if item.get("clinicalAttributeId")
        },
    )

    clinical_rows = [_clinical_row(row) for row in raw_clinical_rows]
    clinical_frame = pd.DataFrame(clinical_rows)
    clinical_frame["clinical_row_idx"] = clinical_frame.index.astype(int)
    clinical_frame.to_csv(metadata_dir / "clinical.csv", index=False)

    pathways = _load_hallmark_pathways(Path(args.hallmark_gmt))
    hallmark_genes = sorted({gene.upper() for genes in pathways.values() for gene in genes})
    gene_map, missing_genes = _fetch_gene_map(hallmark_genes, int(args.gene_map_batch_size))
    entrez_to_symbol = {entrez: symbol for symbol, entrez in gene_map.items()}

    expression_frame = _build_expression_matrix(
        str(args.molecular_profile_id),
        str(args.sample_list_id),
        sorted(entrez_to_symbol.keys()),
        int(args.expression_batch_size),
        entrez_to_symbol,
    )
    expression_frame.to_csv(preprocessed_dir / "expression_matrix.csv")

    pathway_frame = _build_pathway_matrix(expression_frame, pathways)
    pathway_matrix_path = preprocessed_dir / "rna_seq_pathways.csv"
    pathway_frame.to_csv(pathway_matrix_path)

    feature_list_path = pathway_matrix_path.with_suffix(".features.csv")
    with feature_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature"])
        for feature in pathway_frame.columns:
            writer.writerow([feature])

    for patient_barcode, row in pathway_frame.iterrows():
        tensor = torch.tensor(row.to_numpy(dtype="float32"))
        torch.save(tensor, genomics_dir / f"{patient_barcode}.pt")

    metadata = {
        "representation": "hallmark_pathways",
        "num_patients": int(len(pathway_frame)),
        "num_features": int(len(pathway_frame.columns)),
        "molecular_profile_id": str(args.molecular_profile_id),
        "sample_list_id": str(args.sample_list_id),
        "hallmark_gmt": str(Path(args.hallmark_gmt).resolve()),
        "matrix_output": str(pathway_matrix_path.resolve()),
        "feature_list_path": str(feature_list_path.resolve()),
        "feature_construction": "mean_expression_per_hallmark_gene_set",
        "cohort_note": "METABRIC uses microarray expression; this mirrors the repository's TCGA pathway construction rather than full GSVA.",
    }
    write_json(genomics_dir / "metadata.json", metadata)

    clinical_lookup = clinical_frame.set_index("patient_barcode")
    crosswalk_rows: list[dict[str, Any]] = []
    excluded_short_followup = 0
    missing_survival = 0
    for patient_barcode in sorted(set(pathway_frame.index) & set(clinical_lookup.index)):
        clinical_row = clinical_lookup.loc[patient_barcode]
        survival_time = clinical_row["days_to_death"] if clinical_row["days_to_death"] != "" else clinical_row["days_to_last_followup"]
        survival_time_value = float(survival_time) if survival_time not in ("", None) else None
        vital_status = str(clinical_row["vital_status"]).strip()
        _status_text, event_observed = _normalize_status(vital_status)
        if survival_time_value is None or event_observed is None:
            missing_survival += 1
            continue
        label = _binary_label_at_horizon(event_observed, survival_time_value, float(args.horizon_days))
        if label is None:
            excluded_short_followup += 1
            continue
        crosswalk_rows.append(
            {
                "patient_barcode": patient_barcode,
                "genomics_path": str((genomics_dir / f"{patient_barcode}.pt").resolve()),
                "clinical_row_idx": int(clinical_row["clinical_row_idx"]),
                "label": int(label),
                "survival_time": round(float(survival_time_value), 4),
                "event_observed": int(event_observed),
            }
        )

    crosswalk_path = Path(args.crosswalk_output)
    _write_csv(crosswalk_path, crosswalk_rows)
    write_json(
        metadata_dir / "download_summary.json",
        {
            "study_id": STUDY_ID,
            "patient_count_raw_clinical": len(raw_clinical_rows),
            "patient_count_expression": int(len(expression_frame)),
            "patient_count_pathways": int(len(pathway_frame)),
            "patient_count_crosswalk": len(crosswalk_rows),
            "missing_hallmark_gene_symbols": missing_genes,
            "missing_hallmark_gene_count": len(missing_genes),
            "excluded_short_followup_count": excluded_short_followup,
            "missing_survival_count": missing_survival,
            "endpoint_proxy": "5-year overall survival label with OS used as a proxy for PFI",
            "cohort_note": "METABRIC is evaluated without vision; endpoint and assay mismatch relative to TCGA PFI/RNA-seq must be documented.",
        },
    )
    print(
        json.dumps(
            {
                "patient_count_raw_clinical": len(raw_clinical_rows),
                "patient_count_expression": int(len(expression_frame)),
                "patient_count_crosswalk": len(crosswalk_rows),
                "missing_hallmark_gene_count": len(missing_genes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
