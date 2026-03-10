from __future__ import annotations

"""Ablation runner for verifier checkpoints or artifact directories."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .evaluate import _alignment_summary, _classification_metrics, _load_json, _survival_metrics


def _evaluate_artifact(artifact_path: Path) -> dict[str, Any]:
    artifact = _load_json(artifact_path) or {}
    predictions = artifact.get("predictions", [])
    classification = _classification_metrics(predictions)
    survival = _survival_metrics(predictions)
    return {
        "alignment_summary": _alignment_summary(artifact),
        "auroc_macro": classification.get("auroc_macro"),
        "balanced_accuracy": classification.get("balanced_accuracy"),
        "f1_macro": classification.get("f1_macro"),
        "c_index": survival.get("c_index", survival.get("c_index_message", "")),
    }


def main() -> None:
    """Run ablation evaluation across available verifier subdirectories."""
    parser = argparse.ArgumentParser(description="Evaluate verifier ablation checkpoints")
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    root = Path(args.checkpoint_dir)
    candidates = ["verifier_v_only", "verifier_vc", "verifier_vcg", "verifier_full"]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        artifact_path = root / candidate / "artifact.json"
        if artifact_path.exists():
            metrics = _evaluate_artifact(artifact_path)
            rows.append({"Modalities": candidate, **metrics})
    if len(rows) == 1:
        rows[0]["note"] = "Ablation requires multiple checkpoints. Run verifier with --modalities flag."
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "ablation_results.csv"
    fieldnames = ["Modalities", "auroc_macro", "balanced_accuracy", "f1_macro", "c_index", "alignment_summary", "note"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
