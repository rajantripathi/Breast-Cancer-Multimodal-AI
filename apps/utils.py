from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_sample_cases(sample_dir: str | Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(Path(sample_dir).glob("*.json"))]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def load_json_if_exists(path: str | Path) -> Any | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return load_json(candidate)


def load_text_if_exists(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return candidate.read_text()


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_tcga_assets(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    verifier_root = root / "outputs" / "tcga_verifier"
    evaluation_root = root / "reports" / "tcga_evaluation"
    summary_root = root / "reports"
    asset_paths = {
        "verifier_artifact": verifier_root / "artifact.json",
        "demo_cases": verifier_root / "demo_cases.json",
        "verifier_predictions": verifier_root / "predictions.json",
        "verifier_summary": verifier_root / "summary.json",
        "ablation_v_only": root / "outputs" / "ablation_v_only" / "summary.json",
        "ablation_vc": root / "outputs" / "ablation_vc" / "summary.json",
        "ablation_vg": root / "outputs" / "ablation_vg" / "summary.json",
        "enterprise_metrics": first_existing_path(
            [
                evaluation_root / "enterprise_metrics.json",
                root / "outputs" / "enterprise_metrics.json",
            ]
        ),
        "evaluation_report": first_existing_path(
            [
                evaluation_root / "evaluation_report.txt",
                root / "outputs" / "evaluation_report.txt",
            ]
        ),
        "classification_report": first_existing_path(
            [
                evaluation_root / "classification_report.txt",
                root / "outputs" / "classification_report.txt",
            ]
        ),
        "confusion_matrix": first_existing_path(
            [
                evaluation_root / "confusion_matrix.csv",
                root / "outputs" / "confusion_matrix.csv",
            ]
        ),
        "results_summary": summary_root / "tcga_results_summary.md",
        "sample_cases": root / "outputs" / "sample_case_results.json",
    }
    summary_markdown = load_text_if_exists(asset_paths["results_summary"])
    return {
        "paths": asset_paths,
        "verifier_artifact": load_json_if_exists(asset_paths["verifier_artifact"]),
        "demo_cases": load_json_if_exists(asset_paths["demo_cases"]) or [],
        "verifier_predictions": load_json_if_exists(asset_paths["verifier_predictions"]) or [],
        "verifier_summary": load_json_if_exists(asset_paths["verifier_summary"]),
        "ablation_v_only": load_json_if_exists(asset_paths["ablation_v_only"]),
        "ablation_vc": load_json_if_exists(asset_paths["ablation_vc"]),
        "ablation_vg": load_json_if_exists(asset_paths["ablation_vg"]),
        "enterprise_metrics": load_json_if_exists(asset_paths["enterprise_metrics"]) if asset_paths["enterprise_metrics"] else None,
        "evaluation_report": load_text_if_exists(asset_paths["evaluation_report"]) if asset_paths["evaluation_report"] else None,
        "classification_report": load_text_if_exists(asset_paths["classification_report"]) if asset_paths["classification_report"] else None,
        "confusion_matrix": load_text_if_exists(asset_paths["confusion_matrix"]) if asset_paths["confusion_matrix"] else None,
        "results_summary": summary_markdown,
        "results_snapshot": parse_results_summary(summary_markdown) if summary_markdown else {},
        "sample_case_results": load_json_if_exists(asset_paths["sample_cases"]) or [],
    }


def parse_results_summary(summary_markdown: str) -> dict[str, str]:
    patterns = {
        "vision_embeddings": r"Vision embeddings extracted:\s*(.+)",
        "aligned_patients": r"Aligned patients:\s*(.+)",
        "train_split": r"Train / validation / test:\s*(.+)",
        "slides_total": r"TCGA-BRCA slides:\s*(.+)",
        "rnaseq_total": r"TCGA-BRCA RNA-seq:\s*(.+)",
        "clinical_total": r"TCGA-BRCA clinical rows:\s*(.+)",
        "architecture": r"Architecture:\s*(.+)",
        "evaluation_status": r"Evaluation status:\s*(.+)",
    }
    snapshot: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, summary_markdown)
        if match:
            snapshot[key] = match.group(1).strip()
    return snapshot
