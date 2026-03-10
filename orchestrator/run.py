from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents import EHRAgent, GenomicsAgent, LiteratureAgent, VisionAgent
from config import load_settings
from data.common import flatten_payload, read_json
from training.utils import score_with_prototypes


def _artifact_path(repo_root: Path, modality: str) -> str:
    return str(repo_root / "outputs" / modality / "artifact.json")


def _load_verifier(repo_root: Path) -> dict[str, Any]:
    artifact_path = repo_root / "outputs" / "verifier" / "artifact.json"
    if artifact_path.exists():
        return read_json(artifact_path)
    return {"prototypes": {}, "labels": ["monitor", "high_concern"]}


def run_case(case_payload: dict[str, Any], repo_root: Path | None = None) -> dict[str, Any]:
    sample_id = case_payload["sample_id"]
    root = repo_root or Path(__file__).resolve().parents[1]
    agents = [
        VisionAgent(_artifact_path(root, "vision")),
        EHRAgent(_artifact_path(root, "ehr")),
        GenomicsAgent(_artifact_path(root, "genomics")),
        LiteratureAgent(_artifact_path(root, "literature")),
    ]
    predictions = [agent.predict(sample_id, case_payload.get(agent.modality, {})) for agent in agents]
    verifier = _load_verifier(root)
    verifier_features = flatten_payload(case_payload) + " " + " ".join(
        f"{pred.modality}_{pred.predicted_label}" for pred in predictions
    )
    verifier_scores = score_with_prototypes(verifier_features, verifier.get("prototypes", {}))
    fused_label = max(verifier_scores, key=verifier_scores.get) if verifier_scores else "monitor"
    return {
        "sample_id": sample_id,
        "fused_label": fused_label,
        "agent_predictions": [pred.__dict__ for pred in predictions],
        "verifier_scores": verifier_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal fusion")
    parser.add_argument("--case-path", default=None)
    parser.add_argument("--all-sample-cases", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    repo_root = settings.repo_root
    case_paths = [Path(args.case_path)] if args.case_path else sorted((repo_root / "sample_cases").glob("*.json"))
    if not args.all_sample_cases:
        case_paths = case_paths[:1]

    results = [run_case(read_json(case_path), repo_root=repo_root) for case_path in case_paths]
    out_path = repo_root / "outputs" / "fused_predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
