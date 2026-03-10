from __future__ import annotations

from zipfile import ZipFile

from config import load_settings
from data.common import flatten_payload, read_json, read_jsonl, write_jsonl


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


def _build_views(parsed: dict[str, str]) -> list[tuple[str, str]]:
    base_bits = {
        "sample_id": parsed["sample_id"],
        "background_tissue": parsed["background_tissue"],
        "abnormality_class": parsed["abnormality_class"],
        "severity": parsed.get("severity", "N"),
        "x": parsed.get("x", ""),
        "y": parsed.get("y", ""),
        "radius": parsed.get("radius", ""),
    }
    label = parsed["label"]
    views = [
        ("screening_view", flatten_payload(base_bits)),
        (
            "diagnostic_summary",
            f"{parsed['sample_id']} tissue {parsed['background_tissue']} finding {parsed['abnormality_class']} label {label}",
        ),
        (
            "radiology_note",
            f"mammography {label} background {parsed['background_tissue']} lesion {parsed['abnormality_class']}",
        ),
    ]
    if label != "normal":
        views.extend(
            [
                (
                    "lesion_geometry",
                    f"lesion center {parsed.get('x', '')} {parsed.get('y', '')} radius {parsed.get('radius', '')} class {parsed['abnormality_class']}",
                ),
                (
                    "followup_note",
                    f"{label} mass pattern {parsed['abnormality_class']} on {parsed['background_tissue']} tissue",
                ),
                (
                    "diagnosis_prompt",
                    f"predict whether lesion is benign or malignant from class {parsed['abnormality_class']} and severity {parsed.get('severity', '')}",
                ),
            ]
        )
    else:
        views.extend(
            [
                ("normal_screening", f"screening mammogram normal with {parsed['background_tissue']} background tissue"),
                ("screening_recall", f"no suspicious lesion detected on {parsed['background_tissue']} background tissue"),
                ("bilateral_view", f"paired screening film likely normal study {parsed['sample_id']}"),
            ]
        )
    return views


def main() -> None:
    """Prepare a text dataset and extraction manifest for vision workflows."""
    settings = load_settings()
    out_dir = settings.processed_data_root / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    feature_manifest = []
    archive_path = settings.raw_data_root / "vision" / "mias-mammography.zip"
    tcga_manifest_path = settings.raw_data_root / "vision" / "tcga_brca_manifest.jsonl"
    benchmark_manifest_path = settings.raw_data_root / "vision" / "breakhis_manifest.jsonl"
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
                for variant, text in _build_views(parsed):
                    rows.append(
                        {
                            "sample_id": f"{parsed['sample_id']}::{variant}",
                            "label": parsed["label"],
                            "text": text,
                            "metadata": {"source": str(archive_path), "image_name": image_name, "variant": variant},
                        }
                    )
                feature_manifest.append(
                    {
                        "sample_id": parsed["sample_id"],
                        "source_path": f"{archive_path}::{image_name}",
                        "label": parsed["label"],
                        "payload_hint": parsed["abnormality_class"],
                        "metadata": {"dataset": "mias", "image_name": image_name},
                    }
                )
    elif tcga_manifest_path.exists():
        feature_manifest = read_jsonl(tcga_manifest_path)
        for row in feature_manifest:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "text": f"slide manifest sample {row['sample_id']} label {row['label']} source {row['source_path']}",
                    "metadata": row.get("metadata", {}),
                }
            )
        if benchmark_manifest_path.exists():
            for row in read_jsonl(benchmark_manifest_path):
                feature_manifest.append(row)
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "text": f"benchmark image sample {row['sample_id']} label {row['label']} source {row['source_path']}",
                        "metadata": row.get("metadata", {}),
                    }
                )
    else:
        for case_path in sorted((settings.repo_root / "sample_cases").glob("*.json")):
            case = read_json(case_path)
            label = "malignant" if "malignant" in case["sample_id"] else ("benign" if "benign" in case["sample_id"] else "normal")
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "label": label,
                    "text": flatten_payload(case["vision"]),
                    "metadata": {"source": str(case_path)},
                }
            )
            feature_manifest.append(
                {
                    "sample_id": case["sample_id"],
                    "source_path": str(case_path),
                    "label": label,
                    "payload_hint": flatten_payload(case["vision"]),
                    "metadata": {"dataset": "sample_cases", "source": str(case_path)},
                }
            )
    write_jsonl(out_dir / "dataset.jsonl", rows)
    write_jsonl(out_dir / "feature_manifest.jsonl", feature_manifest)
    print(f"vision processed dataset initialized at {out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()
