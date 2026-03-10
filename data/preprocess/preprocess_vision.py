from __future__ import annotations

from zipfile import ZipFile

from config import load_settings
from data.common import flatten_payload, read_json, write_jsonl


def _parse_info_line(line: str) -> dict[str, str] | None:
    parts = line.strip().split()
    if not parts or not parts[0].startswith("mdb"):
        return None
    row = {"sample_id": parts[0], "background_tissue": parts[1], "abnormality_class": parts[2]}
    if parts[2] == "NORM":
        row["label"] = "normal"
        row["severity"] = "N"
        return row
    row["severity"] = parts[3]
    row["label"] = "malignant" if parts[3] == "M" else "benign"
    if len(parts) >= 7:
        row["x"] = parts[4]
        row["y"] = parts[5]
        row["radius"] = parts[6]
    return row


def main() -> None:
    settings = load_settings()
    out_dir = settings.processed_data_root / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    archive_path = settings.raw_data_root / "vision" / "mias-mammography.zip"
    if archive_path.exists():
        with ZipFile(archive_path) as archive:
            info_text = archive.read("Info.txt").decode("utf-8", errors="ignore")
            for line in info_text.splitlines():
                parsed = _parse_info_line(line)
                if not parsed:
                    continue
                image_name = f"all-mias/{parsed['sample_id']}.pgm"
                if image_name not in archive.namelist():
                    continue
                rows.append(
                    {
                        "sample_id": parsed["sample_id"],
                        "label": parsed["label"],
                        "text": flatten_payload(parsed),
                        "metadata": {"source": str(archive_path), "image_name": image_name},
                    }
                )
    else:
        for case_path in sorted((settings.repo_root / "sample_cases").glob("*.json")):
            case = read_json(case_path)
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "label": "malignant" if "malignant" in case["sample_id"] else ("benign" if "benign" in case["sample_id"] else "normal"),
                    "text": flatten_payload(case["vision"]),
                    "metadata": {"source": str(case_path)},
                }
            )
    write_jsonl(out_dir / "dataset.jsonl", rows)
    print(f"vision processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
