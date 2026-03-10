from __future__ import annotations

import csv

from config import load_settings
from data.common import flatten_payload, read_json, write_jsonl


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "genomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    variant_csv = settings.raw_data_root / "genomics" / "genomics_variants.csv"
    if variant_csv.exists():
        with variant_csv.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "text": flatten_payload(row),
                        "metadata": {"source": str(variant_csv)},
                    }
                )
    else:
        for case_path in sorted((settings.repo_root / "sample_cases").glob("*.json")):
            case = read_json(case_path)
            label = "pathogenic_variant" if "pathogenic" in flatten_payload(case["genomics"]) else "benign_variant"
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "label": label,
                    "text": flatten_payload(case["genomics"]),
                    "metadata": {"source": str(case_path)},
                }
            )

    write_jsonl(out_dir / "dataset.jsonl", rows)
    print(f"genomics processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
