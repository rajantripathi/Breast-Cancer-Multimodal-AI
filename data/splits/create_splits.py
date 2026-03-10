from __future__ import annotations

from config import load_settings
from data.common import read_jsonl, write_json


def _split_ids(sample_ids: list[str]) -> dict[str, list[str]]:
    if len(sample_ids) <= 2:
        return {"train": sample_ids[:1], "val": sample_ids[1:2], "test": sample_ids[1:2]}
    return {
        "train": sample_ids[:-2],
        "val": [sample_ids[-2]],
        "test": [sample_ids[-1]],
    }


def main() -> None:
    settings = load_settings()
    split_root = settings.split_root
    split_root.mkdir(parents=True, exist_ok=True)

    for modality in ("vision", "ehr", "genomics", "literature"):
        rows = read_jsonl(settings.processed_data_root / modality / "dataset.jsonl")
        sample_ids = [row["sample_id"] for row in rows]
        write_json(split_root / f"{modality}_splits.json", _split_ids(sample_ids))

    verifier_ids = ["case_01_benign", "case_02_malignant", "case_03_brca"]
    write_json(split_root / "verifier_splits.json", _split_ids(verifier_ids))
    print(f"split manifests written to {split_root}")


if __name__ == "__main__":
    main()
