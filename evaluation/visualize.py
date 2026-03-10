from __future__ import annotations


def render_text_report(metrics: dict[str, object]) -> str:
    lines = ["Evaluation Report"]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

