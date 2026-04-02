"""
Build breast-level metadata from VinDr-Mammo processed metadata.

Each sample corresponds to one breast (left or right) and groups the two
canonical views for that breast: CC and MLO.
"""

import argparse
from pathlib import Path

import pandas as pd


def normalize_view_name(laterality, view_name):
    laterality = str(laterality or "").strip().lower()
    view_name = str(view_name or "").strip().lower()
    if laterality not in {"l", "r"}:
        return None
    if "cc" in view_name:
        return f"{laterality}cc"
    if "mlo" in view_name:
        return f"{laterality}mlo"
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Exam/image-level metadata.csv")
    parser.add_argument("--output", required=True, help="Breast-level metadata output CSV")
    parser.add_argument("--require-both-views", action="store_true", help="Drop breasts missing either CC or MLO")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.metadata).copy()
    df["view_key"] = df.apply(
        lambda row: normalize_view_name(
            row.get("laterality"),
            row.get("view_position") or row.get("view"),
        ),
        axis=1,
    )
    df = df[df["view_key"].notna()].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid VinDr mammography view rows found in metadata.")

    records = []
    for (study_id, laterality), breast_df in df.groupby(["study_id", "laterality"], dropna=False):
        laterality = str(laterality).strip().lower()
        if laterality not in {"l", "r"}:
            continue

        rows = {row["view_key"]: row for _, row in breast_df.iterrows()}
        cc_key = f"{laterality}cc"
        mlo_key = f"{laterality}mlo"
        cc_row = rows.get(cc_key)
        mlo_row = rows.get(mlo_key)

        if args.require_both_views and (cc_row is None or mlo_row is None):
            continue

        labels = [int(row["label"]) for row in [cc_row, mlo_row] if row is not None]
        if not labels:
            continue

        split_values = [str(row["split"]) for row in [cc_row, mlo_row] if row is not None]
        split = split_values[0]
        if any(value != split for value in split_values):
            raise RuntimeError(f"Split mismatch within study={study_id} laterality={laterality}")

        records.append(
            {
                "study_id": study_id,
                "breast_id": f"{study_id}_{laterality}",
                "laterality": laterality,
                "label": max(labels),
                "split": split,
                "cc_image_id": None if cc_row is None else cc_row["image_id"],
                "mlo_image_id": None if mlo_row is None else mlo_row["image_id"],
                "cc_png_path": None if cc_row is None else cc_row.get("png_path", ""),
                "mlo_png_path": None if mlo_row is None else mlo_row.get("png_path", ""),
            }
        )

    breast_df = pd.DataFrame.from_records(records)
    if breast_df.empty:
        raise RuntimeError("Breast-level metadata build produced zero samples.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    breast_df.to_csv(output_path, index=False)

    print(f"Saved breast-level metadata to {output_path}")
    print(f"Breast samples: {len(breast_df)}")
    print(f"Label distribution: {breast_df['label'].value_counts().to_dict()}")
    print(f"Split distribution: {breast_df['split'].value_counts().to_dict()}")
    complete = ((breast_df['cc_image_id'].notna()) & (breast_df['mlo_image_id'].notna())).sum()
    print(f"Breasts with both views: {int(complete)}")


if __name__ == "__main__":
    main()
