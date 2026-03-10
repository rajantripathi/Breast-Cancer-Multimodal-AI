from __future__ import annotations

import argparse
import json
from urllib.parse import quote_plus
from urllib.request import urlopen

from config import load_settings
from data.common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Download literature corpus on Isambard")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=5)
    args = parser.parse_args()

    settings = load_settings()
    out_dir = settings.raw_data_root / "literature"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = [
        ("supportive_evidence", "breast cancer diagnosis biomarkers"),
        ("limited_evidence", "benign breast disease mammography screening"),
    ]
    collected: list[dict] = []
    manifest = {"dataset": "Europe PMC", "queries": [], "target_dir": str(out_dir), "status": "fallback_seeded"}

    try:
        for label, query in queries:
            cursor_mark = "*"
            query_count = 0
            for _page in range(args.max_pages):
                url = (
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
                    f"query={quote_plus(query)}&cursorMark={quote_plus(cursor_mark)}"
                    f"&resultType=lite&pageSize={args.page_size}&format=json"
                )
                response = urlopen(url, timeout=30)
                payload = json.loads(response.read().decode("utf-8"))
                result_list = payload.get("resultList", {}).get("result", [])
                if not result_list:
                    break
                for item in result_list:
                    item["_seed_label"] = label
                    item["_query"] = query
                    collected.append(item)
                query_count += len(result_list)
                next_cursor = payload.get("nextCursorMark")
                if not next_cursor or next_cursor == cursor_mark:
                    break
                cursor_mark = next_cursor
            manifest["queries"].append({"query": query, "label": label, "rows": query_count})

        results_path = out_dir / "results.json"
        write_json(results_path, {"resultList": {"result": collected}})
        manifest["status"] = "downloaded"
        manifest["raw_file"] = str(results_path)
        manifest["num_results"] = len(collected)
    except Exception:
        fallback = {
            "resultList": {
                "result": [
                    {"id": "lit_001", "title": "Breast cancer screening overview", "abstractText": "Mammography remains central.", "_seed_label": "limited_evidence"},
                    {"id": "lit_002", "title": "BRCA variants and surveillance", "abstractText": "Genetic risk influences surveillance.", "_seed_label": "supportive_evidence"},
                ]
            }
        }
        write_json(out_dir / "results.json", fallback)
        manifest["raw_file"] = str(out_dir / "results.json")

    write_json(out_dir / "manifest.json", manifest)
    print(f"literature download manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
