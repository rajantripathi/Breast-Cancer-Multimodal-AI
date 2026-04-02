"""
Download and prepare VinDr-Mammo dataset for training.

VinDr-Mammo structure after download:
  images/
    {study_id}/{image_id}.dicom
  breast-level_annotations.csv
  finding_annotations.csv

This script:
1. Reads breast-level_annotations.csv
2. Assigns exam-level labels:
   - BI-RADS 1,2 -> normal (0)
   - BI-RADS 3 -> probably benign (0 for screening)
   - BI-RADS 4,5 -> suspicious (1)
3. Converts DICOM to 16-bit PNG, preserving aspect ratio and padding to square
4. Creates train/val/test splits (70/15/15, stratified)
5. Saves metadata CSV with columns:
   study_id, image_id, laterality, view, label, split, png_path

Usage:
  python -m agents.mammography.preprocessing.prepare_vindr \
    --input-dir data/mammography/vindr-mammo/raw \
    --output-dir data/mammography/vindr-mammo/processed \
    --image-size 1536
"""

import argparse
from pathlib import Path
import re

import pandas as pd
import numpy as np

try:
    import pydicom
    from PIL import Image
except ImportError:
    pass  # Allow --help without heavy deps


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VinDr-Mammo dataset")
    parser.add_argument("--input-dir", required=True, help="Raw VinDr-Mammo directory")
    parser.add_argument("--output-dir", required=True, help="Processed output directory")
    parser.add_argument("--image-size", type=int, default=1536, help="Square output size")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def assign_label(birads):
    """Map BI-RADS to binary screening label."""
    text = str(birads).strip().upper()
    match = re.search(r"([1-5])", text)
    if match is None:
        return -1  # exclude
    score = int(match.group(1))
    if score in [1, 2, 3]:
        return 0  # normal / probably benign
    elif score in [4, 5]:
        return 1  # suspicious
    return -1  # exclude


def resolve_raw_image_root(input_dir):
    direct = input_dir / "images"
    nested = input_dir / (
        "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
    )
    nested_images = nested / "images"

    candidates = []
    for root in [direct, nested_images]:
        if root.exists():
            count = sum(1 for _ in root.rglob("*.dicom"))
            candidates.append((count, root))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return direct


def normalize_to_uint16(ds):
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 65535.0).clip(0, 65535).astype(np.uint16)


def resize_and_pad_uint16(image_u16, target_size):
    pil = Image.fromarray(image_u16, mode="I;16")
    width, height = pil.size
    scale = min(target_size / max(width, 1), target_size / max(height, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)

    canvas = np.zeros((target_size, target_size), dtype=np.uint16)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = np.asarray(resized, dtype=np.uint16)
    return canvas


def convert_dicom_to_png(dicom_path, png_path, image_size):
    ds = pydicom.dcmread(dicom_path)
    image_u16 = normalize_to_uint16(ds)
    image_u16 = resize_and_pad_uint16(image_u16, image_size)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_u16, mode="I;16").save(png_path)


def main():
    args = parse_args()
    if pydicom is None or Image is None:
        raise ImportError("pydicom and Pillow are required for VinDr preprocessing")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_root = output_dir / f"png_{args.image_size}"
    png_root.mkdir(parents=True, exist_ok=True)
    image_root = resolve_raw_image_root(input_dir)

    # Read annotations
    ann_path = input_dir / "breast-level_annotations.csv"
    if not ann_path.exists():
        ann_path = input_dir / (
            "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
        ) / "breast-level_annotations.csv"
    if not ann_path.exists():
        print(f"ERROR: {ann_path} not found. Download VinDr-Mammo first.")
        print("Access: https://physionet.org/content/vindr-mammo/1.0.0/")
        return

    df = pd.read_csv(ann_path)
    print(f"Loaded {len(df)} breast-level annotations")

    # Assign labels
    df["label"] = df["breast_birads"].apply(assign_label)
    df = df[df["label"] >= 0].reset_index(drop=True)
    print(f"After filtering: {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Stratified split
    from sklearn.model_selection import train_test_split

    study_ids = df["study_id"].unique()
    train_ids, temp_ids = train_test_split(
        study_ids, test_size=0.3, random_state=args.seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=args.seed
    )

    df["split"] = "train"
    df.loc[df["study_id"].isin(val_ids), "split"] = "val"
    df.loc[df["study_id"].isin(test_ids), "split"] = "test"

    print(f"Splits: {df['split'].value_counts().to_dict()}")

    png_paths = []
    converted = 0
    missing = 0
    skipped_corrupt = 0
    for row in df.itertuples(index=False):
        dicom_path = image_root / row.study_id / f"{row.image_id}.dicom"
        png_path = png_root / row.study_id / f"{row.image_id}.png"
        if dicom_path.exists():
            try:
                if not png_path.exists():
                    convert_dicom_to_png(dicom_path, png_path, args.image_size)
                png_paths.append(str(png_path))
                converted += 1
            except Exception:
                png_paths.append("")
                skipped_corrupt += 1
        else:
            png_paths.append("")
            missing += 1

    df["png_path"] = png_paths
    print(
        "PNG conversion complete: "
        f"converted={converted}, missing={missing}, skipped_corrupt={skipped_corrupt}"
    )

    meta_path = output_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
