from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import urlretrieve

from config import load_settings
from data.common import write_json


def _seed_fallback_csv(target: Path) -> None:
    rows = [
        ["sample_id", "age", "tumor_size", "family_history", "label"],
        ["ehr_001", "47", "1.8", "0", "low_risk"],
        ["ehr_002", "62", "3.1", "1", "high_risk"],
        ["ehr_003", "39", "2.4", "1", "high_risk"],
        ["ehr_004", "55", "1.2", "0", "low_risk"],
    ]
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download clinical baseline data on Isambard")
    parser.add_argument(
        "--source-url",
        default="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data",
    )
    args = parser.parse_args()

    settings = load_settings()
    out_dir = settings.raw_data_root / "ehr"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_file = out_dir / "ehr_source.data"
    manifest = {"dataset": args.source_url, "target_dir": str(out_dir), "status": "fallback_seeded"}

    try:
        urlretrieve(args.source_url, raw_file)
        manifest["status"] = "downloaded"
        manifest["raw_file"] = str(raw_file)
    except Exception:
        fallback = out_dir / "ehr_fallback.csv"
        _seed_fallback_csv(fallback)
        manifest["fallback_file"] = str(fallback)

    write_json(out_dir / "manifest.json", manifest)
    print(f"ehr download manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
