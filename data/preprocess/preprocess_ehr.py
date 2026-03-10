from __future__ import annotations

import csv

from config import load_settings
from data.common import flatten_payload, read_json, write_jsonl


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "ehr"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    raw_file = settings.raw_data_root / "ehr" / "ehr_source.data"
    if raw_file.exists():
        with raw_file.open() as handle:
            reader = csv.reader(handle)
            for raw_row in reader:
                if not raw_row or "?" in raw_row:
                    continue
                sample_id = raw_row[0]
                recurrence = raw_row[1]
                time_months = raw_row[2]
                features = {
                    "sample_id": sample_id,
                    "recurrence_status": recurrence,
                    "time_months": time_months,
                    "radius_mean": raw_row[3],
                    "texture_mean": raw_row[4],
                    "perimeter_mean": raw_row[5],
                    "area_mean": raw_row[6],
                }
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": "high_risk" if recurrence == "R" else "low_risk",
                        "text": flatten_payload(features),
                        "metadata": {"source": str(raw_file)},
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
