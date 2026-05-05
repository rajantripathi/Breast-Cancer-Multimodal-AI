"""
Prepare CBIS-DDSM as a train-only auxiliary mammography dataset.

This script normalizes CBIS-DDSM metadata into the same image-level contract
used by the VinDr screener pipeline:

  study_id, image_id, laterality, view, label, split, png_path, raw_path, dataset_source

Implementation notes:
- CBIS-DDSM is treated as an auxiliary training-only source.
- Rows are deduplicated to one image per study/view pair.
- Benign and malignant pathology are both mapped to positive screening labels,
  because CBIS-DDSM is lesion-enriched rather than population-screening data.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import pydicom
    from PIL import Image
except ImportError:
    pass


VIEW_ALIASES = {
    "CC": "cc",
    "MLO": "mlo",
    "LEFT_CC": "cc",
    "LEFT_MLO": "mlo",
    "RIGHT_CC": "cc",
    "RIGHT_MLO": "mlo",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM as auxiliary screener data")
    parser.add_argument("--input-dir", required=True, help="Root directory of CBIS-DDSM raw data")
    parser.add_argument("--output-dir", required=True, help="Processed output directory")
    parser.add_argument(
        "--metadata-csv",
        action="append",
        default=[],
        help="Optional metadata CSVs. If omitted, the script discovers *description*.csv under input-dir.",
    )
    parser.add_argument("--image-size", type=int, default=1536, help="Square output size")
    return parser.parse_args()


def discover_metadata_csvs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.csv") if "description" in path.name.lower())


def normalize_to_uint16(ds):
    arr = ds.pixel_array.astype("float32")
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 65535.0).clip(0, 65535).astype("uint16")


def resize_and_pad_uint16(image_u16, target_size):
    pil = Image.fromarray(image_u16, mode="I;16")
    width, height = pil.size
    scale = min(target_size / max(width, 1), target_size / max(height, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)

    import numpy as np

    canvas = np.zeros((target_size, target_size), dtype="uint16")
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def convert_dicom_to_png(dicom_path: Path, png_path: Path, image_size: int):
    ds = pydicom.dcmread(dicom_path)
    image_u16 = normalize_to_uint16(ds)
    image_u16 = resize_and_pad_uint16(image_u16, image_size)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_u16, mode="I;16").save(png_path)


def infer_laterality(row: pd.Series, path_text: str) -> str | None:
    direct = str(row.get("left or right breast") or row.get("laterality") or "").strip().upper()
    if direct in {"LEFT", "RIGHT"}:
        return "l" if direct == "LEFT" else "r"
    text = path_text.upper()
    if "LEFT" in text:
        return "l"
    if "RIGHT" in text:
        return "r"
    return None


def infer_view(row: pd.Series, path_text: str) -> str | None:
    direct = str(row.get("image view") or row.get("view") or "").strip().upper()
    if direct in VIEW_ALIASES:
        return VIEW_ALIASES[direct]
    text = path_text.upper()
    if "CC" in text:
        return "cc"
    if "MLO" in text:
        return "mlo"
    return None


def infer_study_id(row: pd.Series, path_text: str) -> str:
    for key in ("patient_id", "patient id", "case_id", "case id", "study_id"):
        value = row.get(key)
        if pd.notna(value) and str(value).strip():
            return str(value).strip().replace(" ", "_")
    match = re.search(r"(P_\d+)", path_text.upper())
    if match:
        return match.group(1)
    return Path(path_text).parent.name or Path(path_text).stem


def assign_aux_label(row: pd.Series) -> int:
    pathology = str(row.get("pathology") or "").strip().lower()
    if "malignant" in pathology or "benign" in pathology:
        return 1
    assessment = str(row.get("assessment") or row.get("birads") or "").strip().lower()
    if any(token in assessment for token in ("4", "5", "suspicious")):
        return 1
    return -1


def resolve_dicom_path(input_dir: Path, raw_value: str) -> Path | None:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return None
    search_roots = [input_dir, input_dir / "idc" / "cbis_ddsm"]
    candidates = [
        root / raw_text
        for root in search_roots
    ] + [
        root / f"{raw_text}.dcm"
        for root in search_roots
    ] + [
        root / raw_text.replace("CBIS-DDSM/", "")
        for root in search_roots
    ] + [
        root / f"{raw_text.replace('CBIS-DDSM/', '')}.dcm"
        for root in search_roots
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    parts = Path(raw_text).parts
    if len(parts) >= 3:
        top_name, study_uid, series_uid = parts[:3]
        top_candidates = [top_name]
        if match := re.search(r"(P_\d+_(?:LEFT|RIGHT)_(?:CC|MLO)(?:_\d+)?)", top_name.upper()):
            top_candidates.append(f"{match.group(1)}.dcm")

        for root in search_roots:
            for top_candidate in dict.fromkeys(top_candidates):
                series_dirs = [
                    root / top_candidate / study_uid / series_uid,
                    root / top_candidate / study_uid / f"MG_{series_uid}",
                ]
                for series_dir in series_dirs:
                    if series_dir.is_dir():
                        dicoms = sorted(series_dir.glob("*.dcm"))
                        if dicoms:
                            return dicoms[0]

    tail = Path(raw_text).name
    if tail:
        for root in search_roots:
            matches = list(root.rglob(f"{tail}.dcm"))
            if matches:
                return matches[0]
    return None


def main():
    args = parse_args()
    if pydicom is None or Image is None:
        raise ImportError("pydicom and Pillow are required for CBIS-DDSM preprocessing")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_root = output_dir / f"png_{args.image_size}"
    png_root.mkdir(parents=True, exist_ok=True)

    metadata_paths = [Path(item) for item in args.metadata_csv] if args.metadata_csv else discover_metadata_csvs(input_dir)
    if not metadata_paths:
        raise FileNotFoundError("No CBIS-DDSM metadata CSVs found. Pass --metadata-csv explicitly.")

    rows = []
    converted = 0
    missing = 0
    deduped = 0
    seen = set()
    for metadata_path in metadata_paths:
        frame = pd.read_csv(metadata_path)
        for _, row in frame.iterrows():
            image_value = row.get("image file path") or row.get("image_path") or row.get("full mammogram images path")
            raw_path = resolve_dicom_path(input_dir, image_value)
            if raw_path is None:
                missing += 1
                continue

            path_text = str(image_value or raw_path)
            laterality = infer_laterality(row, path_text)
            view = infer_view(row, path_text)
            if laterality is None or view is None:
                continue

            label = assign_aux_label(row)
            if label < 0:
                continue

            study_id = infer_study_id(row, path_text)
            image_id = raw_path.stem
            view_key = (study_id, laterality, view)
            if view_key in seen:
                deduped += 1
                continue
            seen.add(view_key)

            png_path = png_root / study_id / f"{image_id}.png"
            if not png_path.exists():
                convert_dicom_to_png(raw_path, png_path, args.image_size)
                converted += 1

            rows.append(
                {
                    "study_id": study_id,
                    "image_id": image_id,
                    "laterality": laterality,
                    "view": view,
                    "label": label,
                    "split": "train",
                    "png_path": str(png_path),
                    "raw_path": str(raw_path),
                    "dataset_source": "cbis_ddsm",
                }
            )

    if not rows:
        raise RuntimeError("CBIS-DDSM preprocessing produced zero usable view rows.")

    df = pd.DataFrame.from_records(rows)
    meta_path = output_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)

    complete_studies = (
        df.assign(view_key=df["laterality"].astype(str).str.lower() + df["view"].astype(str).str.lower())
        .groupby("study_id")["view_key"]
        .nunique()
        .ge(4)
        .sum()
    )

    print(f"Metadata CSVs: {[str(path) for path in metadata_paths]}")
    print(f"Saved metadata to {meta_path}")
    print(f"Rows saved: {len(df)}")
    print(f"Studies with all four views: {int(complete_studies)}")
    print(f"PNG conversion complete: converted={converted}, missing={missing}, deduped={deduped}")


if __name__ == "__main__":
    main()
