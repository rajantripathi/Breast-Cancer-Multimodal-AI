"""Prepare EMBED for Stage 1 external mammography evaluation.

This converts EMBED's public clinical and metadata tables into the same
image-level metadata contract used by the screener:

  study_id, image_id, laterality, view, label, split, png_path, raw_path, dataset_source

Design choices:
- Defaults to screening exams only.
- Defaults to a screening external-validation label:
  positive if the exam contains a callback assessment (BI-RADS 0 / ``A``) or a
  malignant pathology severity, negative if all findings are benign/negative.
- Prefers 2D images and can optionally fall back to synthetic C-view when 2D is
  absent for a given view.
- Keeps all rows as ``split=external`` and writes a download manifest for the
  selected DICOM objects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.common import write_json


NEGATIVE_ASSESSMENTS = {"N", "B", "P"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EMBED Open Data for external mammography evaluation")
    parser.add_argument("--input-dir", required=True, help="Raw EMBED directory containing tables/ and images/")
    parser.add_argument("--output-dir", required=True, help="Processed output directory")
    parser.add_argument(
        "--clinical-csv",
        default=None,
        help="Optional clinical CSV override. Defaults to tables/EMBED_OpenData_clinical_reduced.csv if present.",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional metadata CSV override. Defaults to tables/EMBED_OpenData_metadata_reduced.csv if present.",
    )
    parser.add_argument(
        "--exam-type",
        choices=["screening", "diagnostic", "all"],
        default="screening",
        help="Which EMBED exam type subset to export.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["recall_or_pathology", "recall", "suspicious_assessment", "pathology_malignant"],
        default="recall_or_pathology",
        help="How to derive the binary external label from EMBED clinical fields.",
    )
    parser.add_argument(
        "--image-types",
        default="2D,C-view",
        help="Comma-separated FinalImageType values to keep, in descending acceptability.",
    )
    parser.add_argument(
        "--preferred-image-type",
        default="2D",
        help="Preferred FinalImageType when multiple images exist for the same exam/laterality/view.",
    )
    parser.add_argument(
        "--allow-cview-fallback",
        action="store_true",
        help="Allow non-preferred image types from --image-types when the preferred type is absent for a view.",
    )
    parser.add_argument(
        "--full-field-only",
        action="store_true",
        help="Keep only rows with spot_mag == 0 or empty.",
    )
    return parser.parse_args()


def _default_table_path(root: Path, reduced_name: str, full_name: str) -> Path:
    reduced = root / "tables" / reduced_name
    full = root / "tables" / full_name
    if reduced.exists():
        return reduced
    if full.exists():
        return full
    raise FileNotFoundError(f"Could not find {reduced_name} or {full_name} under {root / 'tables'}")


def _normalize_assessment(value: str) -> str:
    text = str(value or "").strip().upper()
    if not text or text == "NAN":
        return ""
    return text[0]


def _normalize_side(value: str) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "R", "B"}:
        return text
    return ""


def _normalize_view(value: str) -> str:
    text = str(value or "").strip().upper()
    if "CC" in text:
        return "cc"
    if "MLO" in text:
        return "mlo"
    return ""


def _normalize_laterality(value: str) -> str:
    text = str(value or "").strip().upper()
    if text == "L":
        return "l"
    if text == "R":
        return "r"
    return ""


def _normalize_density(value: str) -> str:
    text = str(value or "").strip()
    if not text or text.upper() == "NAN":
        return ""
    if text in {"1", "2", "3", "4"}:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}[text]
    upper = text.upper()
    if upper in {"A", "B", "C", "D"}:
        return upper
    return ""


def _classify_exam_type(value: str) -> str:
    text = str(value or "").strip().lower()
    if "screen" in text:
        return "screening"
    if "diag" in text:
        return "diagnostic"
    return "unknown"


def _assessment_positive_codes(exam_type: str, label_mode: str) -> set[str]:
    if label_mode == "suspicious_assessment":
        return {"S", "M", "K"}
    if label_mode == "recall":
        return {"A"} if exam_type == "screening" else {"S", "M", "K"}
    if label_mode == "recall_or_pathology":
        return {"A"} if exam_type == "screening" else {"S", "M", "K"}
    return set()


def _exam_label(exam_type: str, assessments: list[str], malignant_pathology: bool, label_mode: str) -> int:
    positive_assessments = _assessment_positive_codes(exam_type, label_mode)
    if label_mode in {"pathology_malignant", "recall_or_pathology"} and malignant_pathology:
        return 1
    if any(code in positive_assessments for code in assessments):
        return 1
    if any(code in NEGATIVE_ASSESSMENTS for code in assessments):
        return 0
    return -1


def _merge_on_side(metadata: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    merged_frames: list[pd.DataFrame] = []
    for side_code, image_side in (("L", "L"), ("R", "R")):
        clinical_side = clinical[clinical["side_norm"] == side_code]
        metadata_side = metadata[metadata["image_laterality_norm"] == image_side]
        if not clinical_side.empty and not metadata_side.empty:
            merged_frames.append(
                metadata_side.merge(clinical_side, on=["empi_anon", "acc_anon"], how="inner", suffixes=("", "_clinical"))
            )
    bilateral = clinical[clinical["side_norm"].isin({"B", ""})]
    if not bilateral.empty:
        merged_frames.append(
            metadata.merge(bilateral, on=["empi_anon", "acc_anon"], how="inner", suffixes=("", "_clinical"))
        )
    if not merged_frames:
        return pd.DataFrame()
    return pd.concat(merged_frames, ignore_index=True).drop_duplicates()


def _choose_view_rows(frame: pd.DataFrame, preferred_image_type: str, allow_fallback: bool) -> pd.DataFrame:
    priorities = {preferred_image_type: 0}
    if allow_fallback:
        for index, value in enumerate(sorted(frame["FinalImageType"].astype(str).unique())):
            priorities.setdefault(value, index + 1)
    else:
        frame = frame[frame["FinalImageType"] == preferred_image_type]
    if frame.empty:
        return frame
    ranked = frame.assign(_priority=frame["FinalImageType"].map(lambda item: priorities.get(str(item), 100)))
    ranked = ranked.sort_values(
        by=["study_id", "laterality", "view", "_priority", "anon_dicom_path"],
        kind="stable",
    )
    return ranked.drop_duplicates(subset=["study_id", "laterality", "view"], keep="first").drop(columns="_priority")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_csv = Path(args.clinical_csv) if args.clinical_csv else _default_table_path(
        input_dir, "EMBED_OpenData_clinical_reduced.csv", "EMBED_OpenData_clinical.csv"
    )
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else _default_table_path(
        input_dir, "EMBED_OpenData_metadata_reduced.csv", "EMBED_OpenData_metadata.csv"
    )

    clinical = pd.read_csv(clinical_csv, dtype=str).fillna("")
    metadata = pd.read_csv(metadata_csv, dtype=str).fillna("")

    clinical["exam_type"] = clinical["desc"].map(_classify_exam_type)
    if args.exam_type != "all":
        clinical = clinical[clinical["exam_type"] == args.exam_type].copy()

    clinical["assessment_norm"] = clinical["asses"].map(_normalize_assessment)
    clinical["side_norm"] = clinical["side"].map(_normalize_side)
    clinical["density_norm"] = clinical["tissueden"].map(_normalize_density)
    clinical["path_malignant"] = clinical["path_severity"].astype(str).isin({"0", "1"})

    metadata["image_laterality_norm"] = metadata["ImageLateralityFinal"].map(lambda value: str(value or "").strip().upper())
    metadata["laterality"] = metadata["ImageLateralityFinal"].map(_normalize_laterality)
    metadata["view"] = metadata["ViewPosition"].map(_normalize_view)
    metadata["FinalImageType"] = metadata["FinalImageType"].astype(str).str.strip()

    allowed_image_types = [item.strip() for item in str(args.image_types).split(",") if item.strip()]
    metadata = metadata[metadata["FinalImageType"].isin(allowed_image_types)].copy()
    metadata = metadata[metadata["view"].isin({"cc", "mlo"})]
    metadata = metadata[metadata["laterality"].isin({"l", "r"})]
    if args.full_field_only:
        metadata = metadata[metadata["spot_mag"].astype(str).str.strip().isin({"", "0"})].copy()

    merged = _merge_on_side(metadata, clinical)
    if merged.empty:
        raise RuntimeError("EMBED clinical/metadata merge produced zero rows")

    exam_rows = []
    for (empi_anon, acc_anon), exam_df in merged.groupby(["empi_anon", "acc_anon"], dropna=False):
        exam_type_values = [value for value in exam_df["exam_type"].tolist() if value and value != "unknown"]
        exam_type = exam_type_values[0] if exam_type_values else "unknown"
        assessments = [value for value in exam_df["assessment_norm"].tolist() if value]
        malignant_pathology = bool(exam_df["path_malignant"].any())
        label = _exam_label(exam_type, assessments, malignant_pathology, args.label_mode)
        if label < 0:
            continue
        density_values = [value for value in exam_df["density_norm"].tolist() if value]
        density_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
        exam_density = max(density_values, key=lambda item: density_rank.get(item, -1), default="")
        exam_rows.append(
            {
                "empi_anon": empi_anon,
                "acc_anon": acc_anon,
                "study_id": acc_anon,
                "label": int(label),
                "label_assessment": int(any(code in _assessment_positive_codes(exam_type, args.label_mode) for code in assessments)),
                "label_pathology": int(malignant_pathology),
                "exam_type": exam_type,
                "exam_density": exam_density,
            }
        )
    exam_summary = pd.DataFrame.from_records(exam_rows)
    if exam_summary.empty:
        raise RuntimeError("EMBED preparation produced zero labeled exams after applying the requested filters")

    image_rows = merged.drop(columns=["exam_type"], errors="ignore").merge(
        exam_summary,
        on=["empi_anon", "acc_anon"],
        how="inner",
    )
    image_rows["anon_dicom_path"] = image_rows["anon_dicom_path"].astype(str).str.strip().str.lstrip("/")
    image_rows["image_id"] = image_rows["anon_dicom_path"].map(lambda value: Path(value).stem)
    image_rows["raw_path"] = image_rows["anon_dicom_path"].map(lambda value: str(input_dir / "images" / value))
    image_rows["breast_density"] = image_rows["exam_density"]
    image_rows["split"] = "external"
    image_rows["dataset_source"] = "embed"
    image_rows["png_path"] = ""

    selected = _choose_view_rows(
        image_rows[
            [
                "study_id",
                "image_id",
                "laterality",
                "view",
                "label",
                "split",
                "png_path",
                "raw_path",
                "dataset_source",
                "breast_density",
                "exam_density",
                "exam_type",
                "label_assessment",
                "label_pathology",
                "FinalImageType",
                "anon_dicom_path",
            ]
        ].copy(),
        preferred_image_type=str(args.preferred_image_type),
        allow_fallback=bool(args.allow_cview_fallback),
    )
    if selected.empty:
        raise RuntimeError("EMBED preparation produced zero image rows after view/image-type selection")

    metadata_out = output_dir / "metadata.csv"
    selected.drop(columns=["FinalImageType"]).to_csv(metadata_out, index=False)

    manifest_path = output_dir / "download_manifest.txt"
    manifest_path.write_text("\n".join(selected["anon_dicom_path"].tolist()) + "\n")

    four_view_exam_count = (
        selected.assign(view_key=selected["laterality"] + selected["view"])
        .groupby("study_id")["view_key"]
        .nunique()
        .eq(4)
        .sum()
    )
    summary = {
        "exam_type": args.exam_type,
        "label_mode": args.label_mode,
        "allowed_image_types": allowed_image_types,
        "preferred_image_type": args.preferred_image_type,
        "allow_cview_fallback": bool(args.allow_cview_fallback),
        "full_field_only": bool(args.full_field_only),
        "clinical_rows": int(len(clinical)),
        "metadata_rows": int(len(metadata)),
        "merged_rows": int(len(merged)),
        "selected_image_rows": int(len(selected)),
        "selected_exam_count": int(selected["study_id"].nunique()),
        "four_view_exam_count": int(four_view_exam_count),
        "label_distribution": {str(key): int(value) for key, value in selected.groupby("study_id")["label"].first().value_counts().to_dict().items()},
        "exam_type_distribution": {str(key): int(value) for key, value in selected.groupby("study_id")["exam_type"].first().value_counts().to_dict().items()},
    }
    write_json(output_dir / "preparation_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
