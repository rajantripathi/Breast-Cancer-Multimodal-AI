from __future__ import annotations

from config import load_settings
from data.common import flatten_payload, read_json, write_jsonl


def _label_from_case(sample_id: str) -> str:
    if "malignant" in sample_id:
        return "malignant"
    if "benign" in sample_id:
        return "benign"
    return "normal"


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_path in sorted((settings.repo_root / "sample_cases").glob("*.json")):
        case = read_json(case_path)
        rows.append(
            {
                "sample_id": case["sample_id"],
                "label": _label_from_case(case["sample_id"]),
                "text": flatten_payload(case["vision"]),
                "metadata": {"source": str(case_path)},
            }
        )
    write_jsonl(out_dir / "dataset.jsonl", rows)
    print(f"vision processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
