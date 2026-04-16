"""
Prepare CMMD (The Chinese Mammography Database) for auxiliary training.

This script:
1. Reads CMMD clinical XLSX metadata
2. Maps benign -> 0 and malignant -> 1 at breast-side level
3. Converts DICOM to 16-bit PNG, preserving aspect ratio and padding to square
4. Creates 70/15/15 stratified study splits
5. Saves metadata CSV with columns:
   study_id, image_id, laterality, view, label, split, png_path, raw_path, dataset_source

Notes:
- For rows where both ID1 and ID2 exist, TCIA stores ID2 as PatientID in DICOM.
- For D2-XXXX rows, CMMD notes only one side is clinically positive; the other
  side should be treated as benign when missing.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    import pydicom
    from PIL import Image
except ImportError:
    pass


VIEW_KEYS = {"CC", "MLO"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CMMD for auxiliary screener training")
    parser.add_argument("--input-dir", required=True, help="Root directory of CMMD raw data")
    parser.add_argument("--output-dir", required=True, help="Processed output directory")
    parser.add_argument("--clinical-xlsx", default=None, help="Optional explicit CMMD clinical XLSX path")
    parser.add_argument("--image-size", type=int, default=224, help="Square output size")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_clinical_xlsx(input_dir: Path) -> Path:
    matches = sorted(input_dir.rglob("*.xlsx"))
    if not matches:
        raise FileNotFoundError("No CMMD clinical XLSX found under input-dir")
    for path in matches:
        if "clinical" in path.name.lower() or "cmmd" in path.name.lower():
            return path
    return matches[0]


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


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_patient_id(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    return text


def infer_patient_id(row: pd.Series) -> str | None:
    normalized = {normalize_colname(col): row[col] for col in row.index}
    for key in ("id2", "patientid", "patient_id", "subject_id", "study_id", "id1"):
        value = normalize_patient_id(normalized.get(key))
        if value:
            return value
    return None


def parse_label_value(value: object) -> int | None:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return None
    if "malig" in text:
        return 1
    if "benign" in text:
        return 0
    if text in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n"}:
        return 0
    return None


def extract_side_labels(row: pd.Series, patient_id: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    normalized = {normalize_colname(col): row[col] for col in row.index}

    for key, value in normalized.items():
        side = None
        if "left" in key:
            side = "l"
        elif "right" in key:
            side = "r"
        if side is None:
            continue

        parsed = parse_label_value(value)
        if parsed is None:
            continue
        if side not in labels:
            labels[side] = parsed
        elif labels[side] != parsed:
            labels[side] = max(labels[side], parsed)

    if patient_id.upper().startswith("D2-"):
        if labels.get("l") == 1 and "r" not in labels:
            labels["r"] = 0
        if labels.get("r") == 1 and "l" not in labels:
            labels["l"] = 0

    return labels


def build_side_label_map(clinical_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    side_map: dict[str, dict[str, int]] = {}
    skipped = 0
    for _, row in clinical_df.iterrows():
        patient_id = infer_patient_id(row)
        if not patient_id:
            skipped += 1
            continue
        side_labels = extract_side_labels(row, patient_id)
        if not side_labels:
            skipped += 1
            continue
        side_map[patient_id] = side_labels
    print(f"Clinical rows with usable side labels: {len(side_map)} (skipped={skipped})")
    return side_map


def discover_dicom_roots(input_dir: Path) -> list[Path]:
    candidates = [
        input_dir / "idc" / "CMMD",
        input_dir / "idc" / "cmmd",
        input_dir / "CMMD",
        input_dir,
    ]
    existing = []
    for root in candidates:
        if root.exists():
            existing.append(root)
    return existing or [input_dir]


def infer_laterality(ds) -> str | None:
    for attr in ("ImageLaterality", "Laterality"):
        value = str(getattr(ds, attr, "") or "").strip().upper()
        if value in {"L", "R"}:
            return value.lower()
    series_desc = str(getattr(ds, "SeriesDescription", "") or "").upper()
    if "LEFT" in series_desc:
        return "l"
    if "RIGHT" in series_desc:
        return "r"
    return None


def infer_view(ds) -> str | None:
    value = str(getattr(ds, "ViewPosition", "") or "").strip().upper()
    if value in VIEW_KEYS:
        return value.lower()
    series_desc = str(getattr(ds, "SeriesDescription", "") or "").upper()
    if "MLO" in series_desc:
        return "mlo"
    if "CC" in series_desc:
        return "cc"
    return None


def index_dicoms(input_dir: Path) -> dict[str, dict[str, Path]]:
    indexed: dict[str, dict[str, Path]] = defaultdict(dict)
    total = 0
    kept = 0
    for root in discover_dicom_roots(input_dir):
        for dicom_path in root.rglob("*.dcm"):
            total += 1
            try:
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            except Exception:
                continue
            patient_id = normalize_patient_id(getattr(ds, "PatientID", ""))
            laterality = infer_laterality(ds)
            view = infer_view(ds)
            if not patient_id or laterality is None or view is None:
                continue
            key = f"{laterality}{view}"
            indexed[patient_id].setdefault(key, dicom_path)
            kept += 1
    print(f"Indexed CMMD DICOMs: total={total}, usable={kept}, patients={len(indexed)}")
    return indexed


def main():
    args = parse_args()
    if pydicom is None or Image is None:
        raise ImportError("pydicom and Pillow are required for CMMD preprocessing")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_root = output_dir / f"png_{args.image_size}"
    png_root.mkdir(parents=True, exist_ok=True)

    clinical_path = Path(args.clinical_xlsx) if args.clinical_xlsx else discover_clinical_xlsx(input_dir)
    clinical_df = pd.read_excel(clinical_path)
    side_label_map = build_side_label_map(clinical_df)
    dicom_index = index_dicoms(input_dir)

    records = []
    converted = 0
    for patient_id, side_labels in side_label_map.items():
        views = dicom_index.get(patient_id, {})
        if not views:
            continue
        for side, label in side_labels.items():
            for view in ("cc", "mlo"):
                dicom_path = views.get(f"{side}{view}")
                if dicom_path is None:
                    continue
                image_id = dicom_path.stem
                png_path = png_root / patient_id / f"{image_id}.png"
                if not png_path.exists():
                    convert_dicom_to_png(dicom_path, png_path, args.image_size)
                    converted += 1
                records.append(
                    {
                        "study_id": patient_id,
                        "image_id": image_id,
                        "laterality": side,
                        "view": view,
                        "label": int(label),
                        "png_path": str(png_path),
                        "raw_path": str(dicom_path),
                        "dataset_source": "cmmd",
                    }
                )

    if not records:
        raise RuntimeError("CMMD preprocessing produced zero usable rows")

    df = pd.DataFrame.from_records(records)
    study_labels = df.groupby("study_id")["label"].max().rename("study_label").reset_index()

    from sklearn.model_selection import train_test_split

    train_ids, temp_ids = train_test_split(
        study_labels["study_id"],
        test_size=0.30,
        random_state=args.seed,
        stratify=study_labels["study_label"],
    )
    temp_labels = study_labels.set_index("study_id").loc[temp_ids, "study_label"]
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_labels,
    )

    df["split"] = "train"
    df.loc[df["study_id"].isin(val_ids), "split"] = "val"
    df.loc[df["study_id"].isin(test_ids), "split"] = "test"

    meta_path = output_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)

    print(f"Clinical source: {clinical_path}")
    print(f"Rows saved: {len(df)}")
    print(f"Unique studies: {df['study_id'].nunique()}")
    print(f"Split distribution: {df['split'].value_counts().to_dict()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"PNG conversion complete: converted={converted}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
