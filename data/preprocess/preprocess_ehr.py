from __future__ import annotations

import csv

from config import load_settings
from data.common import flatten_payload, read_json, write_jsonl


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "ehr"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    fallback_csv = settings.raw_data_root / "ehr" / "ehr_fallback.csv"
    if fallback_csv.exists():
        with fallback_csv.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "text": flatten_payload(row),
                        "metadata": {"source": str(fallback_csv)},
                    }
                )
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
