from __future__ import annotations

import json
from pathlib import Path

from .metrics import label_distribution
from .visualize import render_text_report


def evaluate_predictions(prediction_file: str | Path) -> dict[str, object]:
    predictions = json.loads(Path(prediction_file).read_text())
    fused_labels = [item.get("fused_label", "unknown") for item in predictions]
    agent_labels = []
    for item in predictions:
        for agent_prediction in item.get("agent_predictions", []):
            agent_labels.append(agent_prediction.get("predicted_label", "unknown"))
    metrics = {
        "num_predictions": len(predictions),
        "fused_label_distribution": label_distribution(fused_labels),
        "agent_label_distribution": label_distribution(agent_labels),
    }
    report_path = Path(prediction_file).with_name("evaluation_report.txt")
    report_path.write_text(render_text_report(metrics))
    return metrics


def main() -> None:
    sample_file = Path(__file__).resolve().parents[1] / "outputs" / "fused_predictions.json"
    if not sample_file.exists():
        sample_file.write_text("[]")
    metrics = evaluate_predictions(sample_file)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
