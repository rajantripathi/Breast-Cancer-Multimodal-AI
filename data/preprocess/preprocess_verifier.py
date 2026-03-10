from __future__ import annotations

from itertools import cycle

from config import load_settings
from data.common import flatten_payload, read_jsonl, stable_shuffle, write_jsonl


POSITIVE_LABELS = {
    "vision": {"malignant"},
    "ehr": {"high_risk"},
    "genomics": {"pathogenic_variant"},
    "literature": {"supportive_evidence"},
}


def _partition_rows(rows: list[dict], modality: str) -> tuple[list[dict], list[dict]]:
    positive = [row for row in rows if row["label"] in POSITIVE_LABELS[modality]]
    negative = [row for row in rows if row["label"] not in POSITIVE_LABELS[modality]]
    return stable_shuffle(positive, seed=len(rows) + len(modality)), stable_shuffle(negative, seed=2 * len(rows) + len(modality))


def _bundle_text(parts: dict[str, dict], target_label: str) -> str:
    return flatten_payload(
        {
            "target_label": target_label,
            "vision": {"sample_id": parts["vision"]["sample_id"], "label": parts["vision"]["label"], "text": parts["vision"]["text"]},
            "ehr": {"sample_id": parts["ehr"]["sample_id"], "label": parts["ehr"]["label"], "text": parts["ehr"]["text"]},
            "genomics": {"sample_id": parts["genomics"]["sample_id"], "label": parts["genomics"]["label"], "text": parts["genomics"]["text"]},
            "literature": {"sample_id": parts["literature"]["sample_id"], "label": parts["literature"]["label"], "text": parts["literature"]["text"]},
        }
    )


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "verifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    modality_rows = {modality: read_jsonl(settings.processed_data_root / modality / "dataset.jsonl") for modality in ("vision", "ehr", "genomics", "literature")}
    positive, negative = {}, {}
    for modality, rows in modality_rows.items():
        positive[modality], negative[modality] = _partition_rows(rows, modality)

    target_size = min(
        6000,
        max(
            2000,
            2 * min(
                len(positive["vision"]) + len(negative["vision"]),
                len(positive["ehr"]) + len(negative["ehr"]),
                len(positive["genomics"]) + len(negative["genomics"]),
                len(positive["literature"]) + len(negative["literature"]),
            ),
        ),
    )

    rows: list[dict] = []
    positive_cycles = {modality: cycle(positive[modality] or negative[modality]) for modality in modality_rows}
    negative_cycles = {modality: cycle(negative[modality] or positive[modality]) for modality in modality_rows}

    for index in range(target_size):
        high_concern = index % 2 == 1
        parts: dict[str, dict] = {}
        for modality in ("vision", "ehr", "genomics", "literature"):
            if high_concern:
                parts[modality] = next(positive_cycles[modality]) if modality != "literature" or index % 3 != 0 else next(negative_cycles[modality])
            else:
                parts[modality] = next(negative_cycles[modality]) if modality != "literature" or index % 5 != 0 else next(positive_cycles[modality])

        target_label = "high_concern" if high_concern else "monitor"
        rows.append(
            {
                "sample_id": f"verifier_bundle_{index:05d}",
                "label": target_label,
                "text": _bundle_text(parts, target_label),
                "metadata": {
                    "vision_id": parts["vision"]["sample_id"],
                    "ehr_id": parts["ehr"]["sample_id"],
                    "genomics_id": parts["genomics"]["sample_id"],
                    "literature_id": parts["literature"]["sample_id"],
                },
            }
        )

    write_jsonl(out_dir / "dataset.jsonl", rows)
    print(f"verifier processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
