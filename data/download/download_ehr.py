from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from config import load_settings
from data.common import write_json


DATA_SOURCES = {
    "wpbc": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data",
    "wdbc": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    "breast_cancer": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download clinical baseline data on Isambard")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    out_dir = settings.raw_data_root / "ehr"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_group": "public_clinical_tabular", "target_dir": str(out_dir), "files": {}, "status": "partial"}

    downloaded = 0
    for name, source_url in DATA_SOURCES.items():
        target = out_dir / f"{name}.data"
        entry = {"source_url": source_url, "target_file": str(target), "status": "missing"}
        try:
            if target.exists() and args.skip_existing:
                entry["status"] = "existing"
            else:
                urlretrieve(source_url, target)
                entry["status"] = "downloaded"
            downloaded += 1
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
        manifest["files"][name] = entry

    fallback = out_dir / "ehr_fallback.csv"
    if downloaded == 0:
        fallback.write_text(
            "sample_id,age,tumor_size,family_history,label\n"
            "ehr_001,47,1.8,0,low_risk\n"
            "ehr_002,62,3.1,1,high_risk\n"
            "ehr_003,39,2.4,1,high_risk\n"
            "ehr_004,55,1.2,0,low_risk\n"
        )
        manifest["fallback_file"] = str(fallback)
        manifest["status"] = "fallback_seeded"
    else:
        manifest["status"] = "downloaded"

    write_json(out_dir / "manifest.json", manifest)
    print(f"ehr download manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
