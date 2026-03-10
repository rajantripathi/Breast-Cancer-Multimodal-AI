from __future__ import annotations

import argparse
from pathlib import Path

from config import load_settings
from data.common import write_json, write_jsonl


def _manifest_row(sample_id: str, source_path: str, label: str, dataset_name: str) -> dict[str, object]:
    """Build one manifest entry for downstream feature extraction."""
    return {
        "sample_id": sample_id,
        "source_path": source_path,
        "label": label,
        "payload_hint": f"{dataset_name}:{sample_id}",
        "metadata": {"dataset": dataset_name, "source_path": source_path},
    }


def main() -> None:
    """Write remote-first vision download manifests for TCGA and benchmark datasets."""
    parser = argparse.ArgumentParser(description="Download vision data on Isambard")
    parser.add_argument("--dataset", default="tcga_brca", choices=["tcga_brca", "breakhis", "mias"])
    args = parser.parse_args()

    settings = load_settings()
    out_dir = settings.raw_data_root / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_to_source = {
        "tcga_brca": {
            "status": "manifest_only",
            "notes": "Primary extraction target. Expect slide paths or pre-extracted embeddings to be staged on Isambard scratch.",
            "source_type": "gdc_or_preextracted",
        },
        "breakhis": {
            "status": "manifest_only",
            "notes": "Benchmark smoke path. Populate image paths on Isambard before extraction.",
            "source_type": "benchmark_images",
        },
        "mias": {
            "status": "legacy_manifest_only",
            "notes": "Legacy mammography fallback path retained for local smoke validation.",
            "source_type": "archive",
        },
    }
    dataset_name = args.dataset
    manifest = {
        "dataset": dataset_name,
        "target_dir": str(out_dir),
        "status": dataset_to_source[dataset_name]["status"],
        "notes": dataset_to_source[dataset_name]["notes"],
        "source_type": dataset_to_source[dataset_name]["source_type"],
    }
    if dataset_name == "tcga_brca":
        rows = [
            _manifest_row("tcga_brca_slide_0001", str(Path("/scratch/tcga-brca") / "slide_0001.svs"), "malignant", dataset_name),
            _manifest_row("tcga_brca_slide_0002", str(Path("/scratch/tcga-brca") / "slide_0002.svs"), "benign", dataset_name),
            _manifest_row("tcga_brca_slide_0003", str(Path("/scratch/tcga-brca") / "slide_0003.svs"), "malignant", dataset_name),
        ]
        write_jsonl(out_dir / "tcga_brca_manifest.jsonl", rows)
        manifest["manifest_path"] = str(out_dir / "tcga_brca_manifest.jsonl")
    elif dataset_name == "breakhis":
        rows = [
            _manifest_row("breakhis_img_0001", str(Path("/scratch/breakhis") / "img_0001.png"), "benign", dataset_name),
            _manifest_row("breakhis_img_0002", str(Path("/scratch/breakhis") / "img_0002.png"), "malignant", dataset_name),
        ]
        write_jsonl(out_dir / "breakhis_manifest.jsonl", rows)
        manifest["manifest_path"] = str(out_dir / "breakhis_manifest.jsonl")
    write_json(out_dir / "manifest.json", manifest)
    print(f"vision dataset manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
