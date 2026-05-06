from __future__ import annotations

"""Download public METABRIC clinical metadata via cBioPortal.

This is the Phase 4 external-validation bootstrap for the non-vision cohort.
It intentionally focuses first on the public patient-level clinical attributes
and study metadata that are already confirmed to exist through the API.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import urllib.request


CBIO_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "brca_metabric"
DEFAULT_ATTRIBUTES = [
    "AGE_AT_DIAGNOSIS",
    "ER_IHC",
    "HER2_SNP6",
    "OS_MONTHS",
    "OS_STATUS",
    "PR_STATUS",
    "TUMOR_STAGE",
    "GRADE",
    "VITAL_STATUS",
    "HISTOLOGICAL_SUBTYPE",
]


def _request_json(path: str) -> Any:
    request = urllib.request.Request(f"{CBIO_BASE}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download METABRIC public clinical metadata from cBioPortal")
    parser.add_argument("--output-dir", required=True, help="Root external/metabric directory")
    parser.add_argument(
        "--attributes",
        nargs="*",
        default=DEFAULT_ATTRIBUTES,
        help="Clinical attribute IDs to extract from cBioPortal",
    )
    parser.add_argument("--page-size", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    study = _request_json(f"/studies/{STUDY_ID}")
    attributes = _request_json(f"/studies/{STUDY_ID}/clinical-attributes?projection=SUMMARY")

    selected_attributes = set(str(item).strip() for item in args.attributes)
    attribute_map = {
        item["clinicalAttributeId"]: {
            "displayName": item.get("displayName"),
            "patientAttribute": bool(item.get("patientAttribute")),
            "datatype": item.get("datatype"),
        }
        for item in attributes
        if item.get("clinicalAttributeId") in selected_attributes
    }

    patients: list[dict[str, Any]] = []
    page = 0
    while True:
        batch = _request_json(
            f"/studies/{STUDY_ID}/patients?projection=SUMMARY&pageSize={int(args.page_size)}&pageNumber={page}"
        )
        if not batch:
            break
        patients.extend(batch)
        if len(batch) < int(args.page_size):
            break
        page += 1

    rows: list[dict[str, Any]] = []
    for patient in patients:
        patient_id = str(patient["patientId"])
        clinical = _request_json(f"/studies/{STUDY_ID}/patients/{patient_id}/clinical-data")
        values = {item.get("clinicalAttributeId"): item.get("value") for item in clinical}
        row = {"patient_id": patient_id}
        for attr in selected_attributes:
            row[attr] = values.get(attr, "")
        rows.append(row)

    _write_csv(metadata_dir / "clinical_raw.csv", rows)
    with (metadata_dir / "study.json").open("w", encoding="utf-8") as handle:
        json.dump(study, handle, indent=2)
    with (metadata_dir / "clinical_attributes.json").open("w", encoding="utf-8") as handle:
        json.dump(attribute_map, handle, indent=2)
    print(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "patient_count": len(rows),
                "attributes": sorted(selected_attributes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
