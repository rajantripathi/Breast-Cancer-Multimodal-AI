from __future__ import annotations

from config import load_settings
from data.common import flatten_payload, read_json, write_jsonl


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "literature"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    raw_results = settings.raw_data_root / "literature" / "results.json"
    if raw_results.exists():
        payload = read_json(raw_results)
        for item in payload.get("resultList", {}).get("result", []):
            text = f"{item.get('title', '')} {item.get('abstractText', '')}".strip()
            seed_label = item.get("_seed_label")
            if seed_label:
                label = seed_label
            else:
                label = "supportive_evidence" if "risk" in text.lower() or "cancer" in text.lower() or "biomarker" in text.lower() else "limited_evidence"
            rows.append(
                {
                    "sample_id": item.get("id", "literature_item"),
                    "label": label,
                    "text": text,
                    "metadata": {"source": str(raw_results), "query": item.get("_query", "")},
                }
            )
    else:
        for case_path in sorted((settings.repo_root / "sample_cases").glob("*.json")):
            case = read_json(case_path)
            query = flatten_payload(case["literature"])
            label = "supportive_evidence" if "cancer" in query.lower() or "brca" in query.lower() else "limited_evidence"
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "label": label,
                    "text": query,
                    "metadata": {"source": str(case_path)},
                }
            )

    write_jsonl(out_dir / "dataset.jsonl", rows)
    print(f"literature processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
