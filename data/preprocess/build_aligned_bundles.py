from __future__ import annotations

"""Build patient-level multimodal crosswalks for aligned verifier training."""

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from config import load_settings

BARCODE_PATTERN = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
CLINICAL_BARCODE_COLUMNS = ("bcr_patient_barcode", "patient_id", "submitter_id")


def _pd():
    import pandas as pd

    return pd


def extract_patient_barcode(value: str) -> str | None:
    """Extract a TCGA patient barcode from a string.

    Args:
        value: File name, path, or clinical identifier.

    Returns:
        The first 12-character TCGA patient barcode if found, otherwise `None`.
    """
    match = BARCODE_PATTERN.search(value.upper())
    if match:
        return match.group(1)[:12]
    cleaned = value.strip().upper()
    return cleaned[:12] if cleaned.startswith("TCGA-") and len(cleaned) >= 12 else None


def _embedding_rows(embedding_dir: Path, modality: str):
    pd = _pd()
    rows: list[dict[str, Any]] = []
    for path in sorted(embedding_dir.rglob("*.pt")):
        patient_id = extract_patient_barcode(path.name) or extract_patient_barcode(str(path))
        if patient_id:
            rows.append({"patient_id": patient_id, f"{modality}_path": str(path)})
    return pd.DataFrame(rows).drop_duplicates(subset=["patient_id"]) if rows else pd.DataFrame(columns=["patient_id", f"{modality}_path"])


def _clinical_rows(clinical_csv: Path):
    pd = _pd()
    frame = pd.read_csv(clinical_csv)
    barcode_column = next((column for column in CLINICAL_BARCODE_COLUMNS if column in frame.columns), None)
    if barcode_column is None:
        lowered = {column.lower(): column for column in frame.columns}
        barcode_column = next((lowered[column] for column in CLINICAL_BARCODE_COLUMNS if column in lowered), None)
    if barcode_column is None:
        raise KeyError("clinical CSV must contain one of bcr_patient_barcode, patient_id, or submitter_id")
    extracted = frame[barcode_column].astype(str).map(extract_patient_barcode)
    clinical = frame.copy()
    clinical["patient_id"] = extracted
    clinical["clinical_row_idx"] = clinical.index.astype(int)
    return clinical.dropna(subset=["patient_id"])[["patient_id", "clinical_row_idx"]].drop_duplicates(subset=["patient_id"])


def build_crosswalk(vision_dir: str | Path, genomics_dir: str | Path, clinical_csv: str | Path, output: str | Path):
    """Build and save a patient-level alignment crosswalk.

    Args:
        vision_dir: Directory containing vision `.pt` embeddings.
        genomics_dir: Directory containing genomics `.pt` embeddings.
        clinical_csv: Clinical CSV with a patient barcode column.
        output: Target CSV path for the crosswalk.

    Returns:
        The saved crosswalk as a pandas DataFrame.
    """
    pd = _pd()
    vision = _embedding_rows(Path(vision_dir), "vision")
    genomics = _embedding_rows(Path(genomics_dir), "genomics")
    clinical = _clinical_rows(Path(clinical_csv))

    vision_ids = set(vision["patient_id"])
    genomics_ids = set(genomics["patient_id"])
    clinical_ids = set(clinical["patient_id"])
    aligned_ids = sorted(vision_ids & genomics_ids & clinical_ids)

    vision_only = len(vision_ids - genomics_ids - clinical_ids)
    genomics_only = len(genomics_ids - vision_ids - clinical_ids)
    clinical_only = len(clinical_ids - vision_ids - genomics_ids)
    print(f"vision-only: {vision_only}")
    print(f"genomics-only: {genomics_only}")
    print(f"clinical-only: {clinical_only}")
    print(f"aligned: {len(aligned_ids)}")
    if len(aligned_ids) < 100:
        print("WARNING: aligned sample count < 100; insufficient for reliable training")

    crosswalk = (
        pd.DataFrame({"patient_id": aligned_ids})
        .merge(vision, on="patient_id", how="left")
        .merge(genomics, on="patient_id", how="left")
        .merge(clinical, on="patient_id", how="left")
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(output_path, index=False)
    return crosswalk


def load_crosswalk(path: str | Path):
    """Load a previously saved alignment crosswalk.

    Args:
        path: CSV path produced by `build_crosswalk`.

    Returns:
        Loaded crosswalk DataFrame.
    """
    return _pd().read_csv(path)


def build_tcga_crosswalk(model: str = "uni2"):
    """Build a TCGA-specific crosswalk from the downloaded manifest.

    Args:
        model: Vision embedding model key used for slide embeddings.

    Returns:
        Saved crosswalk DataFrame.
    """
    pd = _pd()
    settings = load_settings()
    manifest_path = settings.repo_root / "data" / "tcga_brca_manifest.csv"
    clinical_csv = settings.repo_root / "data" / "tcga_brca_clinical.csv"
    output_path = settings.repo_root / "data" / "tcga_crosswalk.csv"
    report_path = settings.repo_root / "reports" / "tcga_alignment_report.txt"
    clinical = pd.read_csv(clinical_csv).copy()
    clinical["patient_barcode"] = clinical["bcr_patient_barcode"].astype(str).map(extract_patient_barcode)
    clinical["clinical_row_idx"] = clinical.index.astype(int)

    manifest = pd.read_csv(manifest_path).copy()
    manifest["patient_barcode"] = manifest["patient_barcode"].astype(str).map(extract_patient_barcode)
    vision_root = settings.project_root / "tcga-brca" / "embeddings" / model
    genomics_root = settings.project_root / "tcga-brca" / "genomics"

    aligned_rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        if str(row.has_slide) != "True" or str(row.has_rnaseq) != "True" or str(row.has_clinical) != "True":
            continue
        patient_barcode = str(row.patient_barcode)
        vision_path = vision_root / f"{patient_barcode}.pt"
        genomics_path = genomics_root / f"{patient_barcode}.pt"
        clinical_match = clinical.loc[clinical["patient_barcode"] == patient_barcode]
        if not vision_path.exists() or not genomics_path.exists() or clinical_match.empty:
            continue
        clinical_row = clinical_match.iloc[0]
        aligned_rows.append(
            {
                "patient_barcode": patient_barcode,
                "vision_path": str(vision_path),
                "genomics_path": str(genomics_path),
                "clinical_row_idx": int(clinical_row["clinical_row_idx"]),
                "vital_status": str(clinical_row.get("vital_status", "")),
                "days_to_death": str(clinical_row.get("days_to_death", "")),
                "days_to_last_followup": str(clinical_row.get("days_to_last_followup", "")),
                "er_status": str(clinical_row.get("er_status_by_ihc", "")),
                "her2_status": str(clinical_row.get("her2_status_by_ihc", "")),
            }
        )
    crosswalk = pd.DataFrame(aligned_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(output_path, index=False)

    total = len(manifest)
    aligned = len(crosswalk)
    fraction = (aligned / total * 100.0) if total else 0.0
    report_lines = [
        f"Total manifest patients: {total}",
        f"Aligned patients: {aligned} of {total} total ({fraction:.2f}%)",
        f"Vision root: {vision_root}",
        f"Genomics root: {genomics_root}",
        f"Clinical CSV: {clinical_csv}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Aligned patients: {aligned} of {total} total ({fraction:.2f}%)")
    return crosswalk


def main() -> None:
    """CLI entrypoint for building patient-level crosswalks."""
    parser = argparse.ArgumentParser(description="Build aligned patient bundles from embedding directories and clinical CSV")
    parser.add_argument("--vision-dir")
    parser.add_argument("--genomics-dir")
    parser.add_argument("--clinical-csv")
    parser.add_argument("--output")
    parser.add_argument("--tcga", action="store_true")
    parser.add_argument("--model", default="uni2")
    args = parser.parse_args()
    if args.tcga:
        crosswalk = build_tcga_crosswalk(model=args.model)
    else:
        if not all([args.vision_dir, args.genomics_dir, args.clinical_csv, args.output]):
            raise SystemExit("--vision-dir, --genomics-dir, --clinical-csv, and --output are required unless --tcga is used")
        crosswalk = build_crosswalk(args.vision_dir, args.genomics_dir, args.clinical_csv, args.output)
    print(crosswalk.head().to_csv(index=False).strip() if not crosswalk.empty else "crosswalk empty")


if __name__ == "__main__":
    main()
