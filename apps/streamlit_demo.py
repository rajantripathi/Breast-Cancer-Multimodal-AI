from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from apps.utils import discover_tcga_assets, load_json
except ModuleNotFoundError:
    from utils import discover_tcga_assets, load_json

NAVY = "#1B3A5C"
TEAL = "#0891B2"
GREEN = "#059669"
AMBER = "#D97706"
RED = "#DC2626"
SOFT = "#F4F8FB"

FROZEN_SCIENCE = {
    "model": "Pathway verifier",
    "endpoint": "Progression-Free Interval (PFI)",
    "evaluation": "5-fold stratified cross-validation",
    "cohort": 788,
    "best_configuration": "Vision + Genomics",
    "c_index_mean": 0.517,
    "c_index_std": 0.045,
    "risk_logrank_p": 0.005,
    "fold_c_index": [0.565, 0.520, 0.434, 0.518, 0.549],
    "secondary": {
        "three_year_auroc": 0.617,
        "five_year_auroc": 0.652,
        "balanced_accuracy": 0.657,
    },
    "ablation": {
        "V": (0.534, 0.072),
        "V+C": (0.526, 0.063),
        "V+G": (0.601, 0.046),
        "V+C+G": (0.589, 0.060),
    },
    "training_gpu": "NVIDIA GH200 120GB",
    "genomics_features": 50,
}


def _inject_style() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(8,145,178,0.10), transparent 24%),
                radial-gradient(circle at top left, rgba(27,58,92,0.10), transparent 28%),
                linear-gradient(180deg, #ffffff 0%, {SOFT} 100%);
        }}
        .block-container {{
            max-width: 1320px;
            padding-top: 1.4rem;
            padding-bottom: 2.8rem;
        }}
        .demo-card {{
            background: white;
            border: 1px solid rgba(27,58,92,0.12);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 16px 34px rgba(27,58,92,0.08);
            min-height: 100%;
        }}
        .card-title {{
            color: {NAVY};
            font-weight: 700;
            margin-bottom: 0.75rem;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            font-weight: 800;
            color: white;
            letter-spacing: 0.02em;
        }}
        .placeholder-box {{
            border: 1px dashed rgba(27,58,92,0.28);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
            background: rgba(8,145,178,0.04);
        }}
        .tiny-note {{
            color: #47637F;
            font-size: 0.9rem;
        }}
        h1, h2, h3 {{
            color: {NAVY};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _choose_cases(assets: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if assets["demo_cases"]:
        return assets["demo_cases"], "demo_cases"
    if assets["verifier_predictions"]:
        return assets["verifier_predictions"], "outputs/tcga_verifier/predictions.json"
    legacy_predictions = assets["paths"]["verifier_predictions"].parent.parent / "verifier" / "predictions.json"
    if legacy_predictions.exists():
        return load_json(legacy_predictions), "outputs/verifier/predictions.json"
    if assets["sample_case_results"]:
        return assets["sample_case_results"], "outputs/sample_case_results.json"
    return [], "No prediction artifact available"


def _pick_record(cases: list[dict[str, Any]], selected_id: str) -> dict[str, Any]:
    return next(case for case in cases if case.get("sample_id", case.get("patient_id")) == selected_id)


def _risk_band(score: float) -> tuple[str, str]:
    if score >= 0.67:
        return "HIGH RISK", RED
    if score >= 0.33:
        return "INTERMEDIATE", AMBER
    return "LOW RISK", GREEN


def _recommendation(label: str) -> str:
    if label == "HIGH RISK":
        return "Elevated modeled progression risk. Prioritize multidisciplinary review."
    if label == "INTERMEDIATE":
        return "Intermediate modeled progression risk. Review alongside pathology and genomics context."
    return "Lower modeled progression risk. Use as decision support, not as a standalone clinical order."


def _score(record: dict[str, Any]) -> float:
    if "risk_score" in record:
        return float(record["risk_score"])
    probabilities = record.get("probabilities", {})
    return float(probabilities.get("high_concern", 0.0))


def _confidence(record: dict[str, Any]) -> float:
    if "confidence" in record:
        return float(record["confidence"])
    probabilities = record.get("probabilities", {})
    if probabilities:
        return float(max(probabilities.values()))
    return 0.0


def _metric_text(value: Any) -> str:
    if value is None:
        return "Pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _modality_state(record: dict[str, Any], modality: str) -> tuple[str, float]:
    modality_predictions = record.get("modality_predictions") or {}
    payload = modality_predictions.get(modality, {})
    if payload:
        return str(payload.get("class", "Pending")), float(payload.get("confidence", 0.0) or 0.0)
    if FROZEN_SCIENCE["best_configuration"] == "Vision + Genomics":
        if modality in {"vision", "genomics"}:
            return "Included in best model", 1.0
        if modality == "clinical":
            return "Not used in frozen best model", 0.0
    return "Unavailable in current artifact", 0.0


def _display_contributions(record: dict[str, Any]) -> dict[str, float]:
    contributions = record.get("modality_contributions") or {}
    modality_predictions = record.get("modality_predictions") or {}
    if modality_predictions:
        return contributions or {"vision": 0.333, "clinical": 0.333, "genomics": 0.333}
    return {"vision": 0.5, "clinical": 0.0, "genomics": 0.5}


def _card_start(title: str) -> None:
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)


def _card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _patient_selector(cases: list[dict[str, Any]], key: str) -> str:
    options = [case.get("sample_id", case.get("patient_id", f"record_{idx}")) for idx, case in enumerate(cases)]
    return st.selectbox("Patient selector", options=options, key=key)


def _donut(contributions: dict[str, float]) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Pie(
                labels=[name.title() for name in contributions],
                values=list(contributions.values()),
                hole=0.56,
                marker=dict(colors=[NAVY, TEAL, "#7FB6D9"]),
                textinfo="label+percent",
            )
        ]
    )
    figure.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY),
        showlegend=False,
    )
    return figure


def _pathway_chart(pathways: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=[float(item.get("activation", 0.0)) for item in pathways],
            y=[str(item.get("name", "Pathway")) for item in pathways],
            orientation="h",
            marker_color=TEAL,
        )
    )
    figure.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        xaxis_title="Activation",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY),
    )
    return figure


def _ablation_chart() -> go.Figure:
    values = {
        name: value[0] * 100.0 for name, value in FROZEN_SCIENCE["ablation"].items()
    }
    figure = go.Figure(
        go.Bar(
            x=list(values.keys()),
            y=list(values.values()),
            marker_color=[NAVY, TEAL, "#4B84B0", GREEN],
            text=[f"{value:.1f}" for value in values.values()],
            textposition="outside",
        )
    )
    figure.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="Validation Accuracy (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY),
    )
    return figure


def _render_patient_risk_assessment(assets: dict[str, Any]) -> None:
    cases, source_label = _choose_cases(assets)
    st.header("Patient Risk Assessment")
    if not cases:
        st.warning("No prediction artifact is available locally yet.")
        return

    selected_id = _patient_selector(cases, "risk_patient")
    record = _pick_record(cases, selected_id)
    risk_score = _score(record)
    confidence = _confidence(record)
    risk_label, color = _risk_band(risk_score)

    left, centre, right = st.columns([1.2, 1.1, 1.1])
    try:
        with left:
            _card_start("Clinical Profile")
            clinical = record.get("clinical_summary", {})
            st.markdown(f"**Age:** {clinical.get('age', 'N/A')}")
            st.markdown(f"**Gender:** {clinical.get('gender', 'N/A')}")
            st.markdown(f"**Stage:** {clinical.get('stage', 'N/A')}")
            st.markdown(f"**ER Status:** {clinical.get('er_status', 'N/A')}")
            st.markdown(f"**PR Status:** {clinical.get('pr_status', 'N/A')}")
            st.markdown(f"**HER2 Status:** {clinical.get('her2_status', 'N/A')}")
            st.markdown(f"**Histological Type:** {clinical.get('histological_type', 'N/A')}")
            st.markdown(f"**Vital Status:** {clinical.get('vital_status', 'N/A')}")
            _card_end()
    except Exception as exc:
        st.warning(f"Clinical profile unavailable: {exc}")

    try:
        with centre:
            _card_start("Risk Assessment")
            st.markdown(
                f'<div class="risk-badge" style="background:{color};">{risk_label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"## {risk_score:.4f}")
            st.progress(min(max(confidence, 0.0), 1.0))
            st.caption(f"Confidence: {confidence:.4f}")
            st.info(_recommendation(risk_label))
            _card_end()
    except Exception as exc:
        st.warning(f"Risk assessment unavailable: {exc}")

    try:
        with right:
            _card_start("Best-Model Inputs")
            contributions = _display_contributions(record)
            st.plotly_chart(_donut(contributions), use_container_width=True)
            st.caption("Frozen headline model uses Vision + Genomics. Clinical data remains available for case review, but it is not part of the best-performing PFI CV configuration.")
            _card_end()
    except Exception as exc:
        st.warning(f"Contribution chart unavailable: {exc}")


def _render_multimodal_analysis(assets: dict[str, Any]) -> None:
    cases, source_label = _choose_cases(assets)
    st.header("Multimodal Analysis")
    if not cases:
        st.warning("No prediction artifact is available locally yet.")
        return

    selected_id = _patient_selector(cases, "analysis_patient")
    record = _pick_record(cases, selected_id)

    col1, col2, col3, col4 = st.columns(4)
    try:
        with col1:
            _card_start("🔬 Vision")
            label, confidence = _modality_state(record, "vision")
            st.markdown(f"**Status:** {label}")
            st.progress(confidence)
            st.markdown(
                '<div class="placeholder-box">🔬<br/>Slide visualization available in deployed system</div>',
                unsafe_allow_html=True,
            )
            st.caption("UNI2 foundation model features from this patient's H&E whole-slide image are part of the frozen best-performing PFI model.")
            _card_end()
    except Exception as exc:
        st.warning(f"Vision panel unavailable: {exc}")

    try:
        with col2:
            _card_start("📋 Clinical")
            label, confidence = _modality_state(record, "clinical")
            clinical = record.get("clinical_summary", {})
            st.markdown(f"**Status:** {label}")
            st.progress(confidence)
            st.markdown(f"Age: {clinical.get('age', 'N/A')}")
            st.markdown(f"Stage: {clinical.get('stage', 'N/A')}")
            st.markdown(f"ER: {clinical.get('er_status', 'N/A')}")
            st.markdown(f"PR: {clinical.get('pr_status', 'N/A')}")
            st.markdown(f"HER2: {clinical.get('her2_status', 'N/A')}")
            st.caption("Clinical covariates are shown for physician context. They did not improve the best PFI cross-validation result in this cohort.")
            _card_end()
    except Exception as exc:
        st.warning(f"Clinical panel unavailable: {exc}")

    try:
        with col3:
            _card_start("🧬 Genomics")
            label, confidence = _modality_state(record, "genomics")
            genomics = record.get("genomics_summary", {})
            st.markdown(f"**Status:** {label}")
            st.progress(confidence)
            st.markdown(f"**Molecular subtype:** {genomics.get('molecular_subtype', 'N/A')}")
            st.plotly_chart(_pathway_chart(genomics.get("top_pathways", [])), use_container_width=True)
            st.caption(f"{FROZEN_SCIENCE['genomics_features']} Hallmark pathways analyzed")
            st.caption("Final science run used MSigDB Hallmark pathway scores derived from TCGA RNA-seq.")
            _card_end()
    except Exception as exc:
        st.warning(f"Genomics panel unavailable: {exc}")

    try:
        with col4:
            _card_start("📚 Literature")
            evidence = record.get("literature_evidence", {})
            predicted_label = str(evidence.get("predicted_label", "interpretive_support")).replace("_", " ").title()
            confidence = float(evidence.get("confidence", 0.0) or 0.0)
            st.markdown(f"**Agent assessment:** {predicted_label}")
            st.progress(confidence)
            if evidence.get("query"):
                st.caption(f"Query context: {evidence['query']}")
            for paper in evidence.get("papers", []):
                with st.expander(paper.get("title", "Evidence paper")):
                    st.markdown(f"**{paper.get('title', 'Untitled')}**")
                    st.markdown(f"*{paper.get('journal', 'Unknown journal')}*")
                    st.caption(paper.get("relevance", ""))
            st.caption(evidence.get("note", "Literature agent provides interpretive support alongside the fused multimodal prediction"))
            _card_end()
    except Exception as exc:
        st.warning(f"Literature panel unavailable: {exc}")


def _render_cohort_performance(assets: dict[str, Any]) -> None:
    frozen = FROZEN_SCIENCE
    st.header("Cohort Performance")
    st.caption("Frozen Final Science Metrics | PFI Endpoint | 5-Fold Stratified Cross-Validation")
    st.info("This page presents the frozen final science results for the proposal recording. It is intentionally pinned to the validated PFI cross-validation summary rather than a live training artifact path.")

    top = st.columns(3)
    top[0].metric("C-index", f"{frozen['c_index_mean']:.3f} +/- {frozen['c_index_std']:.3f}")
    top[0].caption("Primary survival ranking metric across 5 stratified folds")
    top[1].metric("Best Configuration", frozen["best_configuration"])
    top[1].caption("Highest-performing modality combination under PFI cross-validation")
    top[2].metric("Cohort", str(frozen["cohort"]))
    top[2].caption("Patient-aligned TCGA-BRCA cases with vision, pathways, and clinical data")

    row2 = st.columns(3)
    row2[0].metric("3yr AUROC", f"{frozen['secondary']['three_year_auroc']:.3f}")
    row2[0].caption("Supplementary time-dependent discrimination at 3 years")
    row2[1].metric("5yr AUROC", f"{frozen['secondary']['five_year_auroc']:.3f}")
    row2[1].caption("Supplementary time-dependent discrimination at 5 years")
    row2[2].metric("Log-rank p", f"{frozen['risk_logrank_p']:.3f}")
    row2[2].caption("Risk-group separation from the calibrated survival analysis")

    details = pd.DataFrame(
        [
            {"field": "Endpoint", "value": frozen["endpoint"]},
            {"field": "Evaluation", "value": frozen["evaluation"]},
            {"field": "Model", "value": frozen["model"]},
            {"field": "Genomics", "value": "50 Hallmark Pathways"},
            {"field": "Training GPU", "value": frozen["training_gpu"]},
        ]
    )
    st.markdown("#### Training Details")
    st.dataframe(details, use_container_width=True, hide_index=True)

    folds = pd.DataFrame(
        {
            "Fold": [1, 2, 3, 4, 5],
            "C-index": frozen["fold_c_index"],
        }
    )
    st.markdown("#### Per-Fold C-index")
    st.dataframe(folds, use_container_width=True, hide_index=True)

    st.markdown("#### Ablation Comparison")
    st.plotly_chart(_ablation_chart(), use_container_width=True)
    st.caption("Simple late fusion with Vision + Genomics is the strongest and most stable configuration in the frozen PFI 5-fold CV analysis.")

    dataset_inventory = pd.DataFrame(
        [
            {"dataset": "TCGA-BRCA aligned cohort", "count": str(frozen["cohort"])},
            {"dataset": "Genomics features", "count": f"{frozen['genomics_features']} Hallmark pathways"},
            {"dataset": "TCGA-BRCA clinical", "count": "1,097"},
            {"dataset": "UNI2 embeddings extracted", "count": "758"},
            {"dataset": "Slides requiring higher-memory hardware", "count": "358"},
        ]
    )
    st.markdown("#### Dataset Inventory")
    st.dataframe(dataset_inventory, use_container_width=True, hide_index=True)

    st.markdown("#### Scientific Context")
    st.markdown(
        """
        - TCGA-BRCA has 86% censoring, making it one of the hardest cancer types for survival prediction.
        - PFI follows the TCGA-CDR recommendation for BRCA; overall survival is not the preferred endpoint for this disease setting.
        """
    )


def _render_system_architecture() -> None:
    st.header("System Architecture")
    st.markdown(
        """
        **Input Layer**  
        Histopathology WSI embeddings + Hallmark pathway genomics + clinical context + biomedical literature support

        **Frozen Headline Model**  
        Pathway verifier with patient-aligned multimodal fusion

        **Fusion Strategy**  
        Vision embeddings, clinical covariates, and genomics pathway tokens are projected into a shared survival modeling stack with Cox loss

        **Risk Output**  
        Progression-free interval risk ranking with cross-validated C-index reporting
        """
    )
    st.info("Literature agent provides interpretation support in the demo. The frozen science result uses Cox survival loss, the PFI endpoint, and 5-fold stratified cross-validation.")

    registry = pd.DataFrame(
        [
            {"Model": "UNI2", "Type": "Vision", "Dim": 1536, "Status": "Active", "Source": "Harvard/Mahmood Lab"},
            {"Model": "CTransPath", "Type": "Vision", "Dim": 768, "Status": "Active", "Source": "Open access"},
            {"Model": "Virchow", "Type": "Vision", "Dim": 1280, "Status": "Approved", "Source": "Paige AI"},
            {"Model": "CONCH", "Type": "Vision-Language", "Dim": 512, "Status": "Pending", "Source": "Harvard/Mahmood Lab"},
        ]
    )
    st.markdown("#### Foundation Model Registry")
    st.dataframe(registry, use_container_width=True, hide_index=True)

    st.markdown("#### Infrastructure Alignment")
    st.markdown(
        """
        - Training: Isambard-AI national supercomputer (NVIDIA GH200 Grace Hopper)
        - Inference target: Lenovo ThinkSystem SR675i + Intel Xeon Scalable
        - Edge deployment: Lenovo ThinkEdge for hospital-local inference
        - Optimization: Intel OpenVINO for INT8 quantized inference
        """
    )

    timeline = pd.DataFrame(
        [
            {"Phase": "Now", "Milestone": "PFI-aligned 5-fold CV baseline with Vision + Hallmark pathways"},
            {"Phase": "Phase 2", "Milestone": "CPTAC-BRCA external validation"},
            {"Phase": "Phase 2", "Milestone": "Federated privacy-preserving training"},
            {"Phase": "Phase 2", "Milestone": "OpenVINO profiling on Intel Xeon"},
        ]
    )
    st.markdown("#### Phase 2 Roadmap")
    st.dataframe(timeline, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Multimodal AI", page_icon="BC", layout="wide", initial_sidebar_state="expanded")
    _inject_style()
    assets = discover_tcga_assets(Path(__file__).resolve().parents[1])

    st.title("Breast Cancer Multimodal AI")
    st.caption("Clinical Decision Support System | PFI Endpoint | 5-Fold CV")

    page = st.sidebar.radio(
        "Navigate",
        ["Patient Risk Assessment", "Multimodal Analysis", "Cohort Performance", "System Architecture"],
    )
    if page == "Patient Risk Assessment":
        _render_patient_risk_assessment(assets)
    elif page == "Multimodal Analysis":
        _render_multimodal_analysis(assets)
    elif page == "Cohort Performance":
        _render_cohort_performance(assets)
    else:
        _render_system_architecture()

    st.markdown("---")
    st.caption("Dr Rajan Tripathi | AI2 Innovation Lab | github.com/rajantripathi/Breast-Cancer-Multimodal-AI")


if __name__ == "__main__":
    main()
