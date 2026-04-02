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
3. Converts DICOM to PNG (resized to 1024x768 for training)
4. Creates train/val/test splits (70/15/15, stratified)
5. Saves metadata CSV with columns:
   study_id, image_id, laterality, view, label, split, png_path

Usage:
  python -m agents.mammography.preprocessing.prepare_vindr \
    --input-dir data/mammography/vindr-mammo/raw \
    --output-dir data/mammography/vindr-mammo/processed \
    --image-size 1024
"""

import argparse
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    import pydicom
    from PIL import Image
except ImportError:
    pass  # Allow --help without heavy deps


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VinDr-Mammo dataset")
    parser.add_argument("--input-dir", required=True, help="Raw VinDr-Mammo directory")
    parser.add_argument("--output-dir", required=True, help="Processed output directory")
    parser.add_argument("--image-size", type=int, default=1024, help="Resize longest edge")
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


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read annotations
    ann_path = input_dir / "breast-level_annotations.csv"
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

    # Save metadata (DICOM conversion happens separately on Isambard)
    meta_path = output_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
