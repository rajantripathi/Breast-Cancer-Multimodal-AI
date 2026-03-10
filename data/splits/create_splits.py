from __future__ import annotations

from collections import defaultdict

from config import load_settings
from data.common import read_jsonl, stable_shuffle, write_json


def _split_rows(rows: list[dict]) -> dict[str, list[str]]:
    if len(rows) <= 2:
        sample_ids = [row["sample_id"] for row in rows]
        return {"train": sample_ids[:1], "val": sample_ids[1:2], "test": sample_ids[1:2]}

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row["sample_id"])

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for label, sample_ids in grouped.items():
        shuffled = stable_shuffle(sample_ids, seed=len(label) + len(sample_ids))
        if len(shuffled) == 1:
            train.extend(shuffled)
            continue
        if len(shuffled) == 2:
            train.append(shuffled[0])
            test.append(shuffled[1])
            continue
        train_cut = max(1, int(len(shuffled) * 0.7))
        val_cut = max(1, int(len(shuffled) * 0.15))
        remaining_for_test = len(shuffled) - train_cut - val_cut
        if remaining_for_test <= 0:
            val_cut = 1
            train_cut = len(shuffled) - 2
        train.extend(shuffled[:train_cut])
        val.extend(shuffled[train_cut : train_cut + val_cut])
        test.extend(shuffled[train_cut + val_cut :])
    return {"train": train, "val": val, "test": test}


def main() -> None:
    settings = load_settings()
    split_root = settings.split_root
    split_root.mkdir(parents=True, exist_ok=True)

    for modality in ("vision", "ehr", "genomics", "literature"):
        rows = read_jsonl(settings.processed_data_root / modality / "dataset.jsonl")
        write_json(split_root / f"{modality}_splits.json", _split_rows(rows))

    verifier_rows = [
        {"sample_id": "case_01_benign", "label": "monitor"},
        {"sample_id": "case_02_malignant", "label": "high_concern"},
        {"sample_id": "case_03_brca", "label": "high_concern"},
    ]
    write_json(split_root / "verifier_splits.json", _split_rows(verifier_rows))
    print(f"split manifests written to {split_root}")


if __name__ == "__main__":
    main()
