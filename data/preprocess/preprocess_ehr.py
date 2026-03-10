from __future__ import annotations

import csv
from typing import Iterable

from config import load_settings
from data.common import flatten_payload, read_json, write_jsonl

try:
    from sklearn.datasets import load_breast_cancer
except ImportError:  # pragma: no cover
    load_breast_cancer = None


def _iter_wpbc_rows(raw_file: str) -> Iterable[dict]:
    with open(raw_file) as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row or "?" in raw_row:
                continue
            yield {
                "sample_id": f"wpbc_{raw_row[0]}",
                "label": "high_risk" if raw_row[1] == "R" else "low_risk",
                "features": {
                    "recurrence_status": raw_row[1],
                    "time_months": raw_row[2],
                    "radius_mean": raw_row[3],
                    "texture_mean": raw_row[4],
                    "perimeter_mean": raw_row[5],
                    "area_mean": raw_row[6],
                },
            }


def _iter_wdbc_rows(raw_file: str) -> Iterable[dict]:
    with open(raw_file) as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            yield {
                "sample_id": f"wdbc_{raw_row[0]}",
                "label": "high_risk" if raw_row[1] == "M" else "low_risk",
                "features": {
                    "diagnosis": raw_row[1],
                    "radius_mean": raw_row[2],
                    "texture_mean": raw_row[3],
                    "perimeter_mean": raw_row[4],
                    "area_mean": raw_row[5],
                    "smoothness_mean": raw_row[6],
                },
            }


def _iter_breast_cancer_rows(raw_file: str) -> Iterable[dict]:
    columns = [
        "class",
        "age",
        "menopause",
        "tumor_size",
        "inv_nodes",
        "node_caps",
        "deg_malig",
        "breast",
        "breast_quad",
        "irradiat",
    ]
    with open(raw_file) as handle:
        reader = csv.reader(handle)
        for index, raw_row in enumerate(reader):
            if not raw_row or "?" in raw_row:
                continue
            features = dict(zip(columns, raw_row))
            yield {
                "sample_id": f"uci_bc_{index:05d}",
                "label": "high_risk" if features["class"] == "recurrence-events" else "low_risk",
                "features": features,
            }


def _iter_sklearn_rows() -> Iterable[dict]:
    if load_breast_cancer is None:
        return []
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame
    rows: list[dict] = []
    for index, row in frame.iterrows():
        rows.append(
            {
                "sample_id": f"sk_bc_{index:05d}",
                "label": "high_risk" if int(row["target"]) == 0 else "low_risk",
                "features": {
                    "mean_radius": row["mean radius"],
                    "mean_texture": row["mean texture"],
                    "mean_perimeter": row["mean perimeter"],
                    "mean_area": row["mean area"],
                    "worst_radius": row["worst radius"],
                    "worst_texture": row["worst texture"],
                },
            }
        )
    return rows


def _build_views(entry: dict) -> list[tuple[str, str]]:
    features = entry["features"]
    label = entry["label"]
    return [
        ("clinical_profile", flatten_payload(features)),
        (
            "risk_summary",
            f"breast cancer tabular profile label {label} values {flatten_payload(features)}",
        ),
    ]


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "ehr"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    sources = [
        ("wpbc.data", _iter_wpbc_rows),
        ("wdbc.data", _iter_wdbc_rows),
        ("breast_cancer.data", _iter_breast_cancer_rows),
    ]
    total_public_rows = 0
    for filename, iterator in sources:
        raw_file = settings.raw_data_root / "ehr" / filename
        if not raw_file.exists():
            continue
        for entry in iterator(str(raw_file)):
            total_public_rows += 1
            for variant, text in _build_views(entry):
                rows.append(
                    {
                        "sample_id": f"{entry['sample_id']}::{variant}",
                        "label": entry["label"],
                        "text": text,
                        "metadata": {"source": str(raw_file), "variant": variant},
                    }
                )

    sklearn_rows = list(_iter_sklearn_rows())
    for entry in sklearn_rows:
        for variant, text in _build_views(entry):
            rows.append(
                {
                    "sample_id": f"{entry['sample_id']}::{variant}",
                    "label": entry["label"],
                    "text": text,
                    "metadata": {"source": "sklearn.datasets.load_breast_cancer", "variant": variant},
                }
            )

    if rows:
        write_jsonl(out_dir / "dataset.jsonl", rows)
    else:
        for case_path in sorted((settings.repo_root / "sample_cases").glob("*.json")):
            case = read_json(case_path)
            label = "high_risk" if case["ehr"].get("family_history") else "low_risk"
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "label": label,
                    "text": flatten_payload(case["ehr"]),
                    "metadata": {"source": str(case_path)},
                }
            )
        write_jsonl(out_dir / "dataset.jsonl", rows)
    print(f"ehr processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
