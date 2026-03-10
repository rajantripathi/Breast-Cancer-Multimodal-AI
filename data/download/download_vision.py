from __future__ import annotations

import argparse
import shutil
import subprocess

from config import load_settings
from data.common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Download vision data on Isambard")
    parser.add_argument("--dataset-slug", default="kmader/mias-mammography")
    args = parser.parse_args()

    settings = load_settings()
    out_dir = settings.raw_data_root / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset_slug,
        "target_dir": str(out_dir),
        "status": "manifest_only",
        "notes": "Download is expected on Isambard using the Kaggle CLI.",
    }

    if shutil.which("kaggle") and settings.kaggle_username and settings.kaggle_key:
        archive_path = out_dir / "dataset.zip"
        command = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            args.dataset_slug,
            "-p",
            str(out_dir),
            "--force",
        ]
        subprocess.run(command, check=True)
        manifest["status"] = "downloaded"
        manifest["archive"] = str(archive_path)

    write_json(out_dir / "manifest.json", manifest)
    print(f"vision download manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
