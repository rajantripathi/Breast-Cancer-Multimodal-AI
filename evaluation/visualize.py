from __future__ import annotations


def render_text_report(metrics: dict[str, object]) -> str:
    lines = ["Evaluation Report"]
    primary = metrics.get("primary_results", {}) if isinstance(metrics.get("primary_results"), dict) else {}
    if primary:
        if "c_index_mean" in primary:
            lines.extend(
                [
                    "Primary Survival Metrics",
                    f"- C-index mean: {primary.get('c_index_mean')}",
                    f"- C-index std: {primary.get('c_index_std')}",
                    f"- AUROC mean: {primary.get('auroc_mean')}",
                    f"- AUROC std: {primary.get('auroc_std')}",
                    "",
                ]
            )
            for key, value in metrics.items():
                if key == "primary_results":
                    continue
                lines.append(f"- {key}: {value}")
            return "\n".join(lines)
        risk_sep = primary.get("risk_group_separation", {}) if isinstance(primary.get("risk_group_separation"), dict) else {}
        lines.extend(
            [
                "Primary Survival Metrics",
                f"- C-index: {primary.get('c_index')}",
                f"- 5yr AUROC: {primary.get('5yr_auroc')}",
                f"- 3yr AUROC: {primary.get('3yr_auroc')}",
                f"- Risk group separation: p={risk_sep.get('logrank_p_value')}",
                "",
                "Binary Classification (secondary)",
            ]
        )
        secondary = metrics.get("binary_classification_secondary", {})
        if isinstance(secondary, dict):
            for key, value in secondary.items():
                lines.append(f"- {key}: {value}")
        lines.append("")
    for key, value in metrics.items():
        if key in {"primary_results", "binary_classification_secondary"}:
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)
