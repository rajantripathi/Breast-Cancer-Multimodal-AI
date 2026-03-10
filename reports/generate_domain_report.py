from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.common import read_jsonl


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _dataset_size(root: Path, modality: str) -> int:
    return len(read_jsonl(root / "data" / "processed" / modality / "dataset.jsonl"))


def main() -> None:
    repo_root = REPO_ROOT
    outputs_root = repo_root / "outputs"
    summaries = {
        modality: _read_json(outputs_root / modality / "summary.json")
        for modality in ("vision", "ehr", "genomics", "literature", "verifier")
        if (outputs_root / modality / "summary.json").exists()
    }
    fused = _read_json(outputs_root / "fused_predictions.json") if (outputs_root / "fused_predictions.json").exists() else []
    evaluation = {}
    if (outputs_root / "evaluation_report.txt").exists():
        evaluation["report_path"] = str(outputs_root / "evaluation_report.txt")

    dataset_sizes = {modality: _dataset_size(repo_root, modality) for modality in ("vision", "ehr", "genomics", "literature", "verifier")}
    lines = [
        "# Domain Expert Review Pack",
        "",
        "## Current Run Summary",
        "",
        f"- Vision dataset rows: {dataset_sizes['vision']}",
        f"- EHR dataset rows: {dataset_sizes['ehr']}",
        f"- Genomics dataset rows: {dataset_sizes['genomics']}",
        f"- Literature dataset rows: {dataset_sizes['literature']}",
        f"- Verifier dataset rows: {dataset_sizes['verifier']}",
        "",
        "## Latest Metrics",
        "",
    ]
    for modality in ("vision", "ehr", "genomics", "literature", "verifier"):
        summary = summaries.get(modality, {})
        if not summary:
            continue
        lines.append(f"### {modality.capitalize()}")
        for key, value in summary.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend(
        [
            "## Example Fused Outputs",
            "",
        ]
    )
    for row in fused[:3]:
        lines.append(f"- `{row['sample_id']}` -> `{row['fused_label']}`")
    lines.extend(
        [
            "",
            "## Known Limitations",
            "",
            "- The four modality agents now train on much larger public datasets, but they are still stronger baselines rather than full domain-specific transformer fine-tuning.",
            "- The verifier is expanded through weakly aligned synthetic/public bundling, not true patient-linked multimodal cohorts.",
            "- These results support technical review and data-quality feedback, not clinical validation.",
            "",
        ]
    )

    target = repo_root / "reports" / "domain_expert_update.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    print(target)


if __name__ == "__main__":
    main()
