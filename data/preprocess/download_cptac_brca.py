from __future__ import annotations

"""Prepare CPTAC-BRCA external-validation manifests.

This script focuses on the public pieces that are already accessible:

- TCIA/pathdb histopathology manifest with direct `wsiimage_url` links
- GDC CPTAC breast RNA-seq manifest

It computes the patient-level overlap needed for external cohort alignment and
writes the derived manifests to disk. It does not claim to provide public
survival labels; that remains a separate cohort limitation.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any
import urllib.request

from data.common import write_json


TCIA_COLLECTION_URL = "https://www.cancerimagingarchive.net/collection/cptac-brca/"
PATHDB_COHORT_CONFIG_URL = "https://pathdb.cancerimagingarchive.net/system/files/collectionmetadata/202401/cohort_builder_01-27-2024.json"
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"


def _request_text(url: str, *, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    if payload is None:
        with urllib.request.urlopen(url, timeout=120) as response:
            return json.load(response)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _parse_collection_subject_count(html: str) -> int | None:
    match = re.search(r"Subjects\s*</[^>]+>\s*<[^>]+>\s*134\s*<", html, flags=re.IGNORECASE)
    if match:
        return 134
    generic = re.search(r"Subjects\s*\|\s*(\d+)", html, flags=re.IGNORECASE)
    if generic:
        return int(generic.group(1))
    return None


def _fetch_pathdb_manifest() -> tuple[dict[str, Any], list[dict[str, str]]]:
    config = _request_json(PATHDB_COHORT_CONFIG_URL)
    csv_url = str(config["DATA_RESOURCE_URL"])
    text = _request_text(csv_url, timeout=120)
    rows = list(csv.DictReader(text.splitlines()))
    cptac_rows = [row for row in rows if row.get("collection") == "CPTAC-BRCA"]
    return {"config_url": PATHDB_COHORT_CONFIG_URL, "csv_url": csv_url}, cptac_rows


def _fetch_gdc_breast_rnaseq_manifest() -> dict[str, Any]:
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.program.name", "value": "CPTAC"}},
                {"op": "=", "content": {"field": "cases.primary_site", "value": "Breast"}},
                {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            ],
        },
        "fields": "file_id,file_name,cases.submitter_id,cases.project.project_id,data_type",
        "format": "JSON",
        "size": 200,
    }
    response = _request_json(GDC_FILES_URL, payload)
    hits = response.get("data", {}).get("hits", [])
    rows: list[dict[str, str]] = []
    for hit in hits:
        for case in hit.get("cases", []):
            patient_id = str(case.get("submitter_id", "")).strip()
            if not patient_id:
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "file_id": str(hit.get("file_id", "")),
                    "file_name": str(hit.get("file_name", "")),
                    "project_id": str(case.get("project", {}).get("project_id", "")),
                }
            )
    return {
        "rows": rows,
        "patient_ids": sorted({row["patient_id"] for row in rows}),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public CPTAC-BRCA manifest/alignment files for Phase 4")
    parser.add_argument("--output-dir", required=True, help="Root external/cptac_brca directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    collection_html = _request_text(TCIA_COLLECTION_URL)
    subject_count = _parse_collection_subject_count(collection_html)
    pathdb_info, pathdb_rows = _fetch_pathdb_manifest()
    gdc_manifest = _fetch_gdc_breast_rnaseq_manifest()

    tcia_patients = sorted({row.get("patient_id", "").strip() for row in pathdb_rows if row.get("patient_id", "").strip()})
    gdc_patients = sorted(gdc_manifest["patient_ids"])
    overlap = sorted(set(tcia_patients) & set(gdc_patients))

    slide_manifest_rows: list[dict[str, Any]] = []
    for row in pathdb_rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id or patient_id not in overlap:
            continue
        slide_manifest_rows.append(
            {
                "patient_id": patient_id,
                "slide_id": str(row.get("slide_id", "")).strip(),
                "camic_id": str(row.get("camic_id", "")).strip(),
                "wsiimage_url": str(row.get("wsiimage_url", "")).strip(),
                "data_format": str(row.get("data_format", "")).strip(),
                "magnification": str(row.get("magnification", "")).strip(),
                "protocol": str(row.get("protocol", "")).strip(),
                "supporting_data_type": str(row.get("supporting_data_type", "")).strip(),
            }
        )

    gdc_overlap_rows = [row for row in gdc_manifest["rows"] if row["patient_id"] in overlap]

    _write_csv(metadata_dir / "tcia_slide_manifest.csv", slide_manifest_rows)
    _write_csv(metadata_dir / "gdc_rnaseq_manifest.csv", gdc_overlap_rows)

    write_json(
        metadata_dir / "cptac_collection_probe.json",
        {
            "collection_url": TCIA_COLLECTION_URL,
            "subject_count_from_collection_page": subject_count,
            "pathdb_config_url": pathdb_info["config_url"],
            "pathdb_csv_url": pathdb_info["csv_url"],
        },
    )
    write_json(
        metadata_dir / "alignment_probe.json",
        {
            "collection_subject_count": subject_count,
            "tcia_pathdb_patient_count": len(tcia_patients),
            "gdc_rnaseq_patient_count": len(gdc_patients),
            "tcia_gdc_overlap_count": len(overlap),
            "status": "ready_for_vg_alignment",
            "notes": [
                "Direct SVS URLs are available through the TCIA/pathdb cohort-builder CSV.",
                "Patient IDs overlap directly between TCIA/pathdb and GDC for the CPTAC breast cohort.",
                "Public survival endpoints were not identified in the sources used here; treat CPTAC as a V+G / alignment-ready cohort until outcome labels are sourced.",
            ],
            "sample_overlap_patient_ids": overlap[:25],
        },
    )
    print(
        json.dumps(
            {
                "tcia_pathdb_patient_count": len(tcia_patients),
                "gdc_rnaseq_patient_count": len(gdc_patients),
                "tcia_gdc_overlap_count": len(overlap),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
