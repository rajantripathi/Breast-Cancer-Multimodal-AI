from __future__ import annotations

"""Download TCGA-BRCA slides, RNA-seq, and clinical data from GDC."""

import argparse
import csv
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import requests
from requests import exceptions as requests_exceptions

from config import load_settings


GDC_API = "https://api.gdc.cancer.gov"
PROJECT_ID = "TCGA-BRCA"
PAGE_SIZE = 200
DOWNLOAD_TIMEOUT_SECONDS = 1800
MAX_API_RETRIES = 4
MAX_GDC_CLIENT_RETRIES = 3
CLINICAL_COLUMNS = [
    "bcr_patient_barcode",
    "age_at_diagnosis",
    "gender",
    "vital_status",
    "days_to_death",
    "days_to_last_followup",
    "tumor_stage",
    "pathologic_stage",
    "er_status_by_ihc",
    "pr_status_by_ihc",
    "her2_status_by_ihc",
    "histological_type",
]


def _barcode_from_value(value: str | None) -> str | None:
    """Extract a TCGA patient barcode.

    Args:
        value: Candidate TCGA identifier.

    Returns:
        The 12-character TCGA patient barcode, or `None`.
    """
    if not value:
        return None
    value = value.strip().upper()
    if value.startswith("TCGA-") and len(value) >= 12:
        return value[:12]
    return None


def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send a JSON POST request to the GDC API.

    Args:
        endpoint: API endpoint name without leading slash.
        payload: Request JSON body.

    Returns:
        Parsed JSON response.
    """
    response = requests.post(f"{GDC_API}/{endpoint}", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def _fetch_paginated(endpoint: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch all pages for a GDC endpoint.

    Args:
        endpoint: API endpoint name.
        payload: Base request payload.

    Returns:
        Aggregated hit records.
    """
    from_index = 1
    hits: list[dict[str, Any]] = []
    while True:
        page_payload = dict(payload)
        page_payload["from"] = from_index
        page_payload["size"] = PAGE_SIZE
        data = _post(endpoint, page_payload)
        page_hits = data.get("data", {}).get("hits", [])
        if not page_hits:
            break
        hits.extend(page_hits)
        pagination = data.get("data", {}).get("pagination", {})
        total = int(pagination.get("total", len(hits)))
        from_index += len(page_hits)
        if len(hits) >= total:
            break
    return hits


def _file_filters(*clauses: dict[str, Any]) -> dict[str, Any]:
    """Build a GDC file filter payload.

    Args:
        clauses: Individual filter clauses.

    Returns:
        Combined logical filter object.
    """
    return {"op": "and", "content": list(clauses)}


def _eq(field: str, value: str) -> dict[str, Any]:
    """Return a GDC equality clause."""
    return {"op": "=", "content": {"field": field, "value": value}}


def _slide_records(max_slides: int | None) -> list[dict[str, Any]]:
    """Fetch TCGA-BRCA diagnostic slide metadata."""
    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "state",
        "data_type",
        "experimental_strategy",
        "cases.submitter_id",
    ]
    records = _fetch_paginated(
        "files",
        {
            "filters": _file_filters(
                _eq("cases.project.project_id", PROJECT_ID),
                _eq("data_type", "Slide Image"),
                _eq("experimental_strategy", "Diagnostic Slide"),
            ),
            "fields": ",".join(fields),
            "format": "JSON",
        },
    )
    return records[:max_slides] if max_slides else records


def _rnaseq_records(max_samples: int | None) -> list[dict[str, Any]]:
    """Fetch TCGA-BRCA RNA-seq count metadata."""
    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "state",
        "workflow_type",
        "data_type",
        "cases.submitter_id",
    ]
    records = _fetch_paginated(
        "files",
        {
            "filters": _file_filters(
                _eq("cases.project.project_id", PROJECT_ID),
                _eq("data_type", "Gene Expression Quantification"),
                _eq("analysis.workflow_type", "STAR - Counts"),
            ),
            "fields": ",".join(fields),
            "format": "JSON",
        },
    )
    return records[:max_samples] if max_samples else records


def _pick_case_submitter_id(record: dict[str, Any]) -> str | None:
    """Extract the TCGA patient barcode from a file record."""
    for case in record.get("cases", []):
        barcode = _barcode_from_value(case.get("submitter_id"))
        if barcode:
            return barcode
    return None


def _sanitize_name(value: str) -> str:
    """Return a filesystem-safe name fragment."""
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _target_path(target_dir: Path, patient_barcode: str, file_uuid: str, file_name: str) -> Path:
    """Build a normalized local target path."""
    return target_dir / f"{patient_barcode}__{file_uuid}__{_sanitize_name(file_name)}"


def _download_via_api(file_uuid: str, target_path: Path) -> None:
    """Download one GDC file via the HTTP data endpoint with retry support."""
    partial_path = target_path.with_suffix(f"{target_path.suffix}.partial")
    for attempt in range(1, MAX_API_RETRIES + 1):
        existing_size = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
        try:
            with requests.get(
                f"{GDC_API}/data/{file_uuid}",
                headers=headers,
                stream=True,
                timeout=(30, DOWNLOAD_TIMEOUT_SECONDS),
            ) as response:
                response.raise_for_status()
                if existing_size and response.status_code != 206:
                    partial_path.unlink(missing_ok=True)
                    existing_size = 0
                mode = "ab" if existing_size and response.status_code == 206 else "wb"
                with partial_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial_path.replace(target_path)
            return
        except (requests_exceptions.ChunkedEncodingError, requests_exceptions.ConnectionError, requests_exceptions.Timeout) as exc:
            print(f"retryable download failure for {file_uuid} on attempt {attempt}: {exc}")
            if attempt == MAX_API_RETRIES:
                raise
            time.sleep(min(30, attempt * 5))
        except requests_exceptions.RequestException:
            raise


def _download_via_gdc_client(file_uuid: str, target_dir: Path) -> Path:
    """Download one GDC file via gdc-client into the target directory."""
    for attempt in range(1, MAX_GDC_CLIENT_RETRIES + 1):
        try:
            subprocess.run(
                ["gdc-client", "download", file_uuid, "-d", str(target_dir)],
                check=True,
            )
            break
        except subprocess.CalledProcessError:
            if attempt == MAX_GDC_CLIENT_RETRIES:
                raise
            time.sleep(min(30, attempt * 5))
    download_dir = target_dir / file_uuid
    files = [path for path in download_dir.iterdir() if path.is_file()]
    if not files:
        raise FileNotFoundError(f"gdc-client downloaded no files for {file_uuid}")
    return files[0]


def _download_records(
    records: list[dict[str, Any]],
    target_dir: Path,
    progress_label: str,
    on_update: Callable[[list[dict[str, str]]], None] | None = None,
) -> list[dict[str, str]]:
    """Download TCGA file records to a local directory.

    Args:
        records: GDC file metadata records.
        target_dir: Download target directory.
        progress_label: Human-readable label for progress output.

    Returns:
        Download manifest rows.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    use_gdc_client = shutil.which("gdc-client") is not None
    rows: list[dict[str, str]] = []
    skipped = 0
    for index, record in enumerate(records, start=1):
        file_uuid = str(record["file_id"])
        file_name = str(record["file_name"])
        patient_barcode = _pick_case_submitter_id(record)
        if not patient_barcode:
            continue
        destination = _target_path(target_dir, patient_barcode, file_uuid, file_name)
        print(f"{progress_label}: starting {index} / {len(records)} patient={patient_barcode} uuid={file_uuid} file={file_name}")
        if not destination.exists():
            try:
                if use_gdc_client:
                    downloaded = _download_via_gdc_client(file_uuid, target_dir)
                    downloaded.replace(destination)
                    download_dir = target_dir / file_uuid
                    if download_dir.exists():
                        shutil.rmtree(download_dir)
                else:
                    _download_via_api(file_uuid, destination)
            except Exception as exc:
                destination.with_suffix(f"{destination.suffix}.partial").unlink(missing_ok=True)
                skipped += 1
                print(f"{progress_label}: SKIP patient={patient_barcode} uuid={file_uuid} error={exc}")
                continue
        rows.append(
            {
                "patient_barcode": patient_barcode,
                "path": str(destination),
                "uuid": file_uuid,
                "file_name": file_name,
            }
        )
        if on_update is not None:
            on_update(rows)
        if index % 10 == 0:
            print(f"{progress_label}: completed {len(rows)} files, skipped {skipped}, processed {index} / {len(records)}")
    print(f"{progress_label}: finished with {len(rows)} downloaded and {skipped} skipped")
    return rows


def _first_non_empty(*values: Any) -> str:
    """Return the first non-empty string value."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            value = value[0] if value else ""
        text = str(value).strip()
        if text:
            return text
    return ""


def _clinical_rows() -> list[dict[str, str]]:
    """Fetch TCGA-BRCA clinical data through the GDC cases endpoint."""
    fields = [
        "submitter_id",
        "demographic.gender",
        "diagnoses.age_at_diagnosis",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.tumor_stage",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.breast_carcinoma_estrogen_receptor_status",
        "diagnoses.breast_carcinoma_progesterone_receptor_status",
        "diagnoses.lab_proc_her2_neu_immunohistochemistry_receptor_status",
        "diagnoses.primary_diagnosis",
    ]
    cases = _fetch_paginated(
        "cases",
        {
            "filters": _file_filters(_eq("project.project_id", PROJECT_ID)),
            "fields": ",".join(fields),
            "format": "JSON",
        },
    )
    rows: list[dict[str, str]] = []
    for case in cases:
        barcode = _barcode_from_value(case.get("submitter_id"))
        if not barcode:
            continue
        demographic = case.get("demographic") or {}
        diagnoses = case.get("diagnoses") or [{}]
        diagnosis = diagnoses[0] if diagnoses else {}
        rows.append(
            {
                "bcr_patient_barcode": barcode,
                "age_at_diagnosis": _first_non_empty(diagnosis.get("age_at_diagnosis")),
                "gender": _first_non_empty(demographic.get("gender")),
                "vital_status": _first_non_empty(diagnosis.get("vital_status")),
                "days_to_death": _first_non_empty(diagnosis.get("days_to_death")),
                "days_to_last_followup": _first_non_empty(diagnosis.get("days_to_last_follow_up")),
                "tumor_stage": _first_non_empty(diagnosis.get("tumor_stage")),
                "pathologic_stage": _first_non_empty(diagnosis.get("ajcc_pathologic_stage")),
                "er_status_by_ihc": _first_non_empty(diagnosis.get("breast_carcinoma_estrogen_receptor_status")),
                "pr_status_by_ihc": _first_non_empty(diagnosis.get("breast_carcinoma_progesterone_receptor_status")),
                "her2_status_by_ihc": _first_non_empty(diagnosis.get("lab_proc_her2_neu_immunohistochemistry_receptor_status")),
                "histological_type": _first_non_empty(diagnosis.get("primary_diagnosis")),
            }
        )
    rows.sort(key=lambda row: row["bcr_patient_barcode"])
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows with the given schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_master_manifest(
    slide_rows: list[dict[str, str]],
    rnaseq_rows: list[dict[str, str]],
    clinical_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Build the patient-level TCGA master manifest."""
    by_patient: dict[str, dict[str, str]] = defaultdict(
        lambda: {
            "patient_barcode": "",
            "slide_path": "",
            "slide_uuid": "",
            "rnaseq_path": "",
            "rnaseq_uuid": "",
            "has_slide": "False",
            "has_rnaseq": "False",
            "has_clinical": "False",
        }
    )
    for row in slide_rows:
        patient = row["patient_barcode"]
        by_patient[patient]["patient_barcode"] = patient
        by_patient[patient]["slide_path"] = row["path"]
        by_patient[patient]["slide_uuid"] = row["uuid"]
        by_patient[patient]["has_slide"] = "True"
    for row in rnaseq_rows:
        patient = row["patient_barcode"]
        by_patient[patient]["patient_barcode"] = patient
        by_patient[patient]["rnaseq_path"] = row["path"]
        by_patient[patient]["rnaseq_uuid"] = row["uuid"]
        by_patient[patient]["has_rnaseq"] = "True"
    for row in clinical_rows:
        patient = row["bcr_patient_barcode"]
        by_patient[patient]["patient_barcode"] = patient
        by_patient[patient]["has_clinical"] = "True"

    fieldnames = [
        "patient_barcode",
        "slide_path",
        "slide_uuid",
        "rnaseq_path",
        "rnaseq_uuid",
        "has_slide",
        "has_rnaseq",
        "has_clinical",
    ]
    rows = [by_patient[patient] for patient in sorted(by_patient)]
    _write_csv(output_path, rows, fieldnames)

    total = len(rows)
    with_slides = sum(row["has_slide"] == "True" for row in rows)
    with_rnaseq = sum(row["has_rnaseq"] == "True" for row in rows)
    with_clinical = sum(row["has_clinical"] == "True" for row in rows)
    aligned = sum(
        row["has_slide"] == "True" and row["has_rnaseq"] == "True" and row["has_clinical"] == "True" for row in rows
    )
    print(f"total patients: {total}")
    print(f"with slides: {with_slides}")
    print(f"with rnaseq: {with_rnaseq}")
    print(f"with clinical: {with_clinical}")
    print(f"with all three: {aligned}")


def main() -> None:
    """Download TCGA-BRCA multimodal assets and build a manifest."""
    parser = argparse.ArgumentParser(description="Download TCGA-BRCA slides, RNA-seq, and clinical data")
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    tcga_root = settings.project_root / "tcga-brca"
    slides_dir = tcga_root / "slides"
    rnaseq_dir = tcga_root / "rnaseq"
    clinical_csv = settings.repo_root / "data" / "tcga_brca_clinical.csv"
    manifest_csv = settings.repo_root / "data" / "tcga_brca_manifest.csv"

    print("querying GDC slide metadata")
    slide_meta = _slide_records(args.max_slides)
    print("querying GDC RNA-seq metadata")
    rnaseq_meta = _rnaseq_records(args.max_samples)
    print("querying GDC clinical metadata")
    clinical_rows = _clinical_rows()
    _write_csv(clinical_csv, clinical_rows, CLINICAL_COLUMNS)

    slide_rows: list[dict[str, str]] = []
    rnaseq_rows: list[dict[str, str]] = []

    def update_manifest(_: list[dict[str, str]] | None = None) -> None:
        _build_master_manifest(slide_rows, rnaseq_rows, clinical_rows, manifest_csv)

    update_manifest()
    slide_rows = _download_records(slide_meta, slides_dir, "slides", on_update=update_manifest)
    update_manifest()
    rnaseq_rows = _download_records(rnaseq_meta, rnaseq_dir, "rnaseq", on_update=update_manifest)
    update_manifest()

    summary = {
        "slides_downloaded": len(slide_rows),
        "rnaseq_downloaded": len(rnaseq_rows),
        "clinical_rows": len(clinical_rows),
        "slides_dir": str(slides_dir),
        "rnaseq_dir": str(rnaseq_dir),
        "clinical_csv": str(clinical_csv),
        "manifest_csv": str(manifest_csv),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
