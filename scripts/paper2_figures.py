"""
Generate publication figures for Paper 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FINAL_RESULTS_PATH = ROOT / "reports" / "paper" / "final_results.json"
KM_DATA_PATH = ROOT / "reports" / "paper" / "km_data.json"
MAMMO_SUMMARY_PATH = ROOT / "outputs" / "mammography" / "summary.json"
OUT_DIR = ROOT / "reports" / "paper2" / "figures"

NAVY = "#1B3A5C"
TEAL = "#0891B2"
GREEN = "#059669"
AMBER = "#D97706"
GRAY = "#6B7280"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def draw_box(ax, xy, width, height, text, color, text_color="white"):
    rect = Rectangle(xy, width, height, facecolor=color, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        wrap=True,
        fontsize=11,
        fontweight="semibold",
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color=GRAY,
        )
    )


def figure1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.05, 0.62), 0.18, 0.18, "Stage 1\nMammography\nScreening", NAVY)
    draw_box(ax, (0.30, 0.62), 0.18, 0.18, "4-view\nConvNeXt-Base\nAUROC 0.741", TEAL)
    draw_box(ax, (0.56, 0.62), 0.18, 0.18, "Routing\nSuspicious cases ->\npathology workup", GREEN)
    draw_box(ax, (0.79, 0.62), 0.16, 0.18, "Stage 2\nMultimodal\nRisk Model", AMBER)

    draw_box(ax, (0.18, 0.18), 0.16, 0.16, "Pathology\nWSI", NAVY, "white")
    draw_box(ax, (0.40, 0.18), 0.16, 0.16, "Genomics\nRNA-seq", TEAL, "white")
    draw_box(ax, (0.62, 0.18), 0.16, 0.16, "Clinical\nCovariates", GREEN, "white")

    ax.text(0.87, 0.25, "CONCH V+C+G\nC-index 0.609\nPFI | n = 1,043", ha="center", va="center", fontsize=11)

    arrow(ax, (0.23, 0.71), (0.30, 0.71))
    arrow(ax, (0.48, 0.71), (0.56, 0.71))
    arrow(ax, (0.74, 0.71), (0.79, 0.71))
    arrow(ax, (0.26, 0.34), (0.84, 0.62))
    arrow(ax, (0.48, 0.34), (0.84, 0.62))
    arrow(ax, (0.70, 0.34), (0.84, 0.62))

    ax.text(0.5, 0.93, "Two-stage breast cancer AI workflow", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.14, 0.54, "Screening", ha="center", color=NAVY, fontsize=11, fontweight="bold")
    ax.text(0.87, 0.54, "Diagnosis + Prognosis", ha="center", color=AMBER, fontsize=11, fontweight="bold")
    save(fig, "figure1_two_stage_architecture.png")


def figure2_mammography_curve() -> None:
    summary = load_json(MAMMO_SUMMARY_PATH)
    auroc = summary["test_auroc"]
    sens90spec = summary["test_sensitivity_at_90_specificity"]
    spec90sens = summary["test_specificity_at_90_sensitivity"]

    # No raw ROC arrays are stored in the repo. Use the saved operating points only.
    fpr = np.array([0.0, 0.10, 1.0 - spec90sens, 1.0])
    tpr = np.array([0.0, sens90spec, 0.90, 1.0])
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = np.maximum.accumulate(tpr[order])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color=NAVY, linewidth=2.2, marker="o", markersize=5)
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Mammography operating characteristic")
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)

    ax.annotate(
        f"AUROC = {auroc:.3f}",
        xy=(0.62, 0.14),
        xycoords="axes fraction",
        color=NAVY,
        fontsize=11,
        fontweight="semibold",
    )
    ax.annotate(
        f"Sensitivity @ 90% specificity = {sens90spec:.3f}",
        xy=(0.33, sens90spec),
        xytext=(0.38, sens90spec + 0.09),
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.2},
        color=TEAL,
    )
    ax.annotate(
        f"Specificity @ 90% sensitivity = {spec90sens:.3f}",
        xy=(1.0 - spec90sens, 0.90),
        xytext=(0.08, 0.96),
        arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 1.2},
        color=GREEN,
    )
    save(fig, "figure2_mammography_performance_curve.png")


def figure3_encoder_comparison(final_results: dict) -> None:
    means = {
        "CONCH": final_results["encoder_comparison"]["conch_ca_vg"]["c_index_mean"],
        "CTransPath": final_results["encoder_comparison"]["ctranspath_ca_vg"]["c_index_mean"],
        "UNI2": final_results["encoder_comparison"]["uni2_ca_vg"]["c_index_mean"],
    }
    labels = list(means.keys())
    values = [means[k] for k in labels]
    colors = [NAVY, TEAL, GREEN]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0.5, 0.62)
    ax.set_ylabel("C-index")
    ax.set_title("Pathology encoder comparison (V+G, cross-attention)")
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom")
    save(fig, "figure3_encoder_comparison.png")


def figure4_ablation(final_results: dict) -> None:
    ablation = final_results["ablation"]
    labels = ["V", "V+C", "V+G", "V+C+G"]
    values = [
        ablation["v_only"]["c_index_mean"],
        ablation["v_c"]["c_index_mean"],
        ablation["v_g"]["c_index_mean"],
        ablation["v_c_g"]["c_index_mean"],
    ]
    colors = [NAVY, TEAL, GREEN, AMBER]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0.5, 0.63)
    ax.set_ylabel("C-index")
    ax.set_title("CONCH ablation study")
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom")
    save(fig, "figure4_ablation.png")


def figure5_km_curves(km_data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    groups = [
        ("Low risk", km_data["low_risk"], NAVY),
        ("Mid risk", km_data["mid_risk"], TEAL),
        ("High risk", km_data["high_risk"], AMBER),
    ]
    for label, group, color in groups:
        times = np.array([0.0] + group["times"])
        survival = np.array([1.0] + group["survival"])
        ax.step(times, survival, where="post", color=color, linewidth=2, label=f"{label} (n={group['n']})")

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days")
    ax.set_ylabel("Progression-free survival probability")
    ax.set_title("Kaplan-Meier risk stratification")
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.04,
        0.12,
        f"log-rank p = {km_data['log_rank_p']:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="semibold",
    )
    save(fig, "figure5_km_curves.png")


def main() -> None:
    set_style()
    final_results = load_json(FINAL_RESULTS_PATH)
    km_data = load_json(KM_DATA_PATH)

    figure1_architecture()
    figure2_mammography_curve()
    figure3_encoder_comparison(final_results)
    figure4_ablation(final_results)
    figure5_km_curves(km_data)
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
