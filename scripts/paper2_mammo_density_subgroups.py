from __future__ import annotations

"""Stage 1 mammography density-subgroup analysis from saved predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def summarize_group(df: pd.DataFrame) -> dict:
    y_true = df["true_label"].to_numpy(dtype=np.int64)
    y_score = df["predicted_probability"].to_numpy(dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    auroc = None
    if positives > 0 and negatives > 0:
        auroc = float(roc_auc_score(y_true, y_score))
    return {
        "n_exams": int(len(df)),
        "n_positive": positives,
        "prevalence": round(float(y_true.mean()), 4) if len(df) else 0.0,
        "mean_score": round(float(y_score.mean()), 4) if len(df) else 0.0,
        "auroc": round(auroc, 4) if auroc is not None else None,
    }


def main() -> None:
    args = parse_args()
    predictions = pd.read_json(Path(args.predictions_json))
    metadata = pd.read_csv(Path(args.metadata_csv))

    if "exam_density" not in metadata.columns:
        raise RuntimeError("metadata.csv does not contain exam_density; rerun prepare_vindr.py first.")

    density_lookup = (
        metadata.loc[:, ["study_id", "exam_density"]]
        .drop_duplicates(subset=["study_id"])
        .assign(exam_density=lambda df: df["exam_density"].fillna("").astype(str).str.strip().str.upper())
    )
    merged = predictions.merge(density_lookup, on="study_id", how="left")
    merged["exam_density"] = merged["exam_density"].replace("", "UNKNOWN").fillna("UNKNOWN")

    overall = summarize_group(merged)
    by_density = {
        density: summarize_group(group.reset_index(drop=True))
        for density, group in merged.groupby("exam_density", sort=True)
    }

    output = {
        "overall": overall,
        "densities_present": sorted(by_density.keys()),
        "by_density": by_density,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
