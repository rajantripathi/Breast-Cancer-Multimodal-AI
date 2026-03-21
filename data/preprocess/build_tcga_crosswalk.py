from __future__ import annotations

"""Build a TCGA patient alignment crosswalk from currently available artifacts."""

import argparse
from pathlib import Path

import pandas as pd

from config import load_settings
from data.preprocess.build_aligned_bundles import extract_patient_barcode

CLINICAL_COLUMNS = [
    "patient_barcode",
    "vision_path",
    "genomics_path",
    "clinical_row_idx",
    "vital_status",
    "days_to_death",
    "days_to_last_followup",
    "er_status_by_ihc",
    "pr_status_by_ihc",
    "her2_status_by_ihc",
]


def _paths_to_frame(root: Path, column_name: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*.pt")):
        barcode = extract_patient_barcode(path.stem)
        if barcode is None:
            continue
        rows.append({"patient_barcode": barcode, column_name: str(path)})
    if not rows:
        return pd.DataFrame(columns=["patient_barcode", column_name])
    return pd.DataFrame(rows).drop_duplicates(subset=["patient_barcode"], keep="first")


def build_tcga_crosswalk() -> pd.DataFrame:
    return build_tcga_crosswalk_with_roots()


def build_tcga_crosswalk_with_roots(
    genomics_root: Path | None = None,
    output_csv: Path | None = None,
    report_path: Path | None = None,
) -> pd.DataFrame:
    settings = load_settings()
    vision_root = settings.project_root / "tcga-brca" / "embeddings" / "uni2"
    genomics_root = genomics_root or (settings.project_root / "tcga-brca" / "genomics")
    clinical_csv = settings.repo_root / "data" / "tcga_brca_clinical.csv"
    output_csv = output_csv or (settings.repo_root / "data" / "tcga_crosswalk.csv")
    report_path = report_path or (settings.repo_root / "reports" / "tcga_alignment_report.txt")

    vision = _paths_to_frame(vision_root, "vision_path")
    genomics = _paths_to_frame(genomics_root, "genomics_path")

    clinical = pd.read_csv(clinical_csv).copy()
    clinical["patient_barcode"] = clinical["bcr_patient_barcode"].astype(str).map(extract_patient_barcode)
    clinical["clinical_row_idx"] = clinical.index.astype(int)
    clinical = clinical.dropna(subset=["patient_barcode"]).drop_duplicates(subset=["patient_barcode"], keep="first")

    aligned = (
        vision.merge(genomics, on="patient_barcode", how="inner")
        .merge(
            clinical[
                [
                    "patient_barcode",
                    "clinical_row_idx",
                    "vital_status",
                    "days_to_death",
                    "days_to_last_followup",
                    "er_status_by_ihc",
                    "pr_status_by_ihc",
                    "her2_status_by_ihc",
                ]
            ],
            on="patient_barcode",
            how="inner",
        )
        .sort_values("patient_barcode")
        .reset_index(drop=True)
    )
    aligned = aligned[CLINICAL_COLUMNS]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_csv, index=False)

    report_lines = [
        f"Vision patients with embeddings: {vision['patient_barcode'].nunique()}",
        f"Genomics patients with tensors: {genomics['patient_barcode'].nunique()}",
        f"Clinical patients with rows: {clinical['patient_barcode'].nunique()}",
        f"Aligned patients: {len(aligned)}",
        f"Vision root: {vision_root}",
        f"Genomics root: {genomics_root}",
        f"Clinical CSV: {clinical_csv}",
        f"Crosswalk CSV: {output_csv}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n")
    print(
        f"Aligned patients: {len(aligned)} "
        f"(vision={vision['patient_barcode'].nunique()}, "
        f"genomics={genomics['patient_barcode'].nunique()}, "
        f"clinical={clinical['patient_barcode'].nunique()})",
        flush=True,
    )
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TCGA patient crosswalk from available embeddings")
    parser.add_argument("--genomics-root", default=None, help="Override the genomics tensor root for alternative representations")
    parser.add_argument("--output-csv", default=None, help="Override the output CSV path")
    parser.add_argument("--report-path", default=None, help="Override the alignment report path")
    args = parser.parse_args()
    genomics_root = Path(args.genomics_root) if args.genomics_root else None
    output_csv = Path(args.output_csv) if args.output_csv else None
    report_path = Path(args.report_path) if args.report_path else None
    build_tcga_crosswalk_with_roots(genomics_root=genomics_root, output_csv=output_csv, report_path=report_path)


if __name__ == "__main__":
    main()
