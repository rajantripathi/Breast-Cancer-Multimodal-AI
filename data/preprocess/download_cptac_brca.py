from __future__ import annotations

"""Probe CPTAC-BRCA external-validation readiness.

This script intentionally starts with metadata discovery and patient-ID
alignment checks before attempting large slide downloads. The current Phase 4
workflow should stop if TCIA imaging metadata cannot be linked reliably to the
genomics/clinical side.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from data.common import write_json


TCIA_COLLECTION_URL = "https://www.cancerimagingarchive.net/collection/cptac-brca/"
TCIA_NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"


def _request_text(url: str, *, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _request_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def _probe_tcia_manifest() -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for endpoint in ("getPatient", "getSeries"):
        url = f"{TCIA_NBIA_BASE}/{endpoint}?Collection=CPTAC-BRCA&format=json"
        try:
            body = _request_text(url)
            attempts.append(
                {
                    "endpoint": endpoint,
                    "url": url,
                    "status": "ok",
                    "content_length": len(body),
                    "preview": body[:200],
                }
            )
        except urllib.error.HTTPError as exc:
            attempts.append(
                {
                    "endpoint": endpoint,
                    "url": url,
                    "status": "http_error",
                    "code": int(exc.code),
                    "reason": str(exc.reason),
                }
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            attempts.append(
                {
                    "endpoint": endpoint,
                    "url": url,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    non_empty = [item for item in attempts if item.get("status") == "ok" and int(item.get("content_length", 0)) > 0]
    return {"attempts": attempts, "usable_manifest": bool(non_empty)}


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
    patient_ids = sorted(
        {
            str(case.get("submitter_id")).strip()
            for hit in hits
            for case in hit.get("cases", [])
            if str(case.get("submitter_id", "")).strip()
        }
    )
    return {
        "hit_count": int(len(hits)),
        "patient_ids": patient_ids,
        "project_ids": sorted(
            {
                str(case.get("project", {}).get("project_id")).strip()
                for hit in hits
                for case in hit.get("cases", [])
                if str(case.get("project", {}).get("project_id", "")).strip()
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe CPTAC-BRCA TCIA/GDC external-validation readiness")
    parser.add_argument("--output-dir", required=True, help="Target root for external CPTAC metadata")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only write metadata probes and stop before any download attempt",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    collection_html = _request_text(TCIA_COLLECTION_URL)
    subject_count = _parse_collection_subject_count(collection_html)
    tcia_probe = _probe_tcia_manifest()
    gdc_manifest = _fetch_gdc_breast_rnaseq_manifest()

    write_json(
        metadata_dir / "cptac_collection_probe.json",
        {
            "collection_url": TCIA_COLLECTION_URL,
            "subject_count_from_collection_page": subject_count,
        },
    )
    write_json(metadata_dir / "tcia_nbia_probe.json", tcia_probe)
    write_json(metadata_dir / "gdc_rnaseq_manifest.json", gdc_manifest)

    alignment_summary = {
        "collection_subject_count": subject_count,
        "gdc_rnaseq_patient_count": int(len(gdc_manifest["patient_ids"])),
        "tcia_patient_manifest_available": bool(tcia_probe["usable_manifest"]),
        "alignment_count": None,
        "status": "blocked" if not tcia_probe["usable_manifest"] else "ready_for_patient_join",
        "blocker": (
            "TCIA NBIA endpoints did not return a usable patient-level manifest for CPTAC-BRCA; "
            "cannot compute patient alignment count programmatically from this environment."
            if not tcia_probe["usable_manifest"]
            else ""
        ),
    }
    write_json(metadata_dir / "alignment_probe.json", alignment_summary)

    if not tcia_probe["usable_manifest"]:
        raise RuntimeError(alignment_summary["blocker"])
    if args.probe_only:
        print(json.dumps(alignment_summary, indent=2))
        return
    raise RuntimeError(
        "CPTAC-BRCA patient manifest is available, but bulk WSI download/tiling is not implemented in this probe script yet."
    )


if __name__ == "__main__":
    main()
