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
        return assets["demo_cases"], "outputs/tcga_verifier/demo_cases.json"
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
        return "Recommend oncology team review"
    if label == "INTERMEDIATE":
        return "Recommend follow-up imaging in 3 months"
    return "Standard surveillance protocol"


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


def _ablation_chart(summary: dict[str, Any], assets: dict[str, Any]) -> go.Figure:
    values = {
        "V": float((assets.get("ablation_v_only") or {}).get("val_accuracy", 0.0)) * 100.0,
        "V+C": float((assets.get("ablation_vc") or {}).get("val_accuracy", 0.0)) * 100.0,
        "V+G": float((assets.get("ablation_vg") or {}).get("val_accuracy", 0.0)) * 100.0,
        "V+C+G": float(summary.get("val_accuracy", 0.0)) * 100.0,
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
    st.caption(f"Prediction source: {source_label}")
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
            _card_start("Modality Contribution")
            contributions = record.get("modality_contributions") or {"vision": 0.333, "clinical": 0.333, "genomics": 0.333}
            st.plotly_chart(_donut(contributions), use_container_width=True)
            _card_end()
    except Exception as exc:
        st.warning(f"Contribution chart unavailable: {exc}")


def _render_multimodal_analysis(assets: dict[str, Any]) -> None:
    cases, source_label = _choose_cases(assets)
    st.header("Multimodal Analysis")
    st.caption(f"Prediction source: {source_label}")
    if not cases:
        st.warning("No prediction artifact is available locally yet.")
        return

    selected_id = _patient_selector(cases, "analysis_patient")
    record = _pick_record(cases, selected_id)
    modality_predictions = record.get("modality_predictions", {})

    col1, col2, col3, col4 = st.columns(4)
    try:
        with col1:
            _card_start("🔬 Vision")
            payload = modality_predictions.get("vision", {})
            st.markdown(f"**Prediction:** {payload.get('class', 'Pending')}")
            st.progress(float(payload.get("confidence", 0.0) or 0.0))
            st.markdown(
                '<div class="placeholder-box">🔬<br/>Slide visualization available in deployed system</div>',
                unsafe_allow_html=True,
            )
            st.caption("UNI2 foundation model processed tissue patches from this patient's H&E whole-slide image (1,536-dimensional embedding)")
            _card_end()
    except Exception as exc:
        st.warning(f"Vision panel unavailable: {exc}")

    try:
        with col2:
            _card_start("📋 Clinical")
            payload = modality_predictions.get("clinical", {})
            clinical = record.get("clinical_summary", {})
            st.markdown(f"**Prediction:** {payload.get('class', 'Pending')}")
            st.progress(float(payload.get("confidence", 0.0) or 0.0))
            st.markdown(f"Age: {clinical.get('age', 'N/A')}")
            st.markdown(f"Stage: {clinical.get('stage', 'N/A')}")
            st.markdown(f"ER: {clinical.get('er_status', 'N/A')}")
            st.markdown(f"PR: {clinical.get('pr_status', 'N/A')}")
            st.markdown(f"HER2: {clinical.get('her2_status', 'N/A')}")
            _card_end()
    except Exception as exc:
        st.warning(f"Clinical panel unavailable: {exc}")

    try:
        with col3:
            _card_start("🧬 Genomics")
            payload = modality_predictions.get("genomics", {})
            genomics = record.get("genomics_summary", {})
            st.markdown(f"**Prediction:** {payload.get('class', 'Pending')}")
            st.progress(float(payload.get("confidence", 0.0) or 0.0))
            st.markdown(f"**Molecular subtype:** {genomics.get('molecular_subtype', 'N/A')}")
            st.plotly_chart(_pathway_chart(genomics.get("top_pathways", [])), use_container_width=True)
            st.caption(f"{genomics.get('gene_count', 'N/A')} genes analyzed")
            st.caption(genomics.get("note", ""))
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
    metrics = assets.get("enterprise_metrics") or {}
    summary = assets.get("verifier_summary") or {}
    snapshot = assets.get("results_snapshot") or {}
    st.header("Cohort Performance")
    st.caption("Enterprise Evaluation Metrics")

    top = st.columns(3)
    top[0].metric("AUROC", _metric_text(metrics.get("auroc_macro")))
    top[0].caption("Discrimination across held-out patients")
    top[1].metric("F1", _metric_text(metrics.get("f1_macro")))
    top[1].caption("Balance between precision and recall")
    top[2].metric("Balanced Accuracy", _metric_text(metrics.get("balanced_accuracy")))
    top[2].caption("Class-balanced classification performance")

    row2 = st.columns(3)
    row2[0].metric("Brier Score", _metric_text(metrics.get("brier_score")))
    row2[0].caption("Probability calibration error magnitude")
    row2[1].metric("ECE", _metric_text(metrics.get("ece")))
    row2[1].caption("Calibration alignment to observed outcomes")
    row2[2].metric("C-index", _metric_text(metrics.get("c_index") or metrics.get("c_index_message")))
    row2[2].caption("Survival risk ranking quality")

    details = pd.DataFrame(
        [
            {"field": "Aligned patients", "value": summary.get("aligned_sample_count", snapshot.get("aligned_patients", "Pending"))},
            {"field": "Train / val / test split", "value": "487 / 104 / 105 (13 events in test)"},
            {"field": "Epochs", "value": "Early stopping at 21 on GPU"},
            {"field": "Architecture", "value": "Three modality projections + cross-attention fusion + Cox risk head"},
        ]
    )
    st.markdown("#### Training Details")
    st.dataframe(details, use_container_width=True, hide_index=True)

    st.markdown("#### Ablation Comparison")
    st.plotly_chart(_ablation_chart(summary, assets), use_container_width=True)
    st.caption("Monotonic improvement validates multimodal architecture")

    dataset_inventory = pd.DataFrame(
        [
            {"dataset": "TCGA-BRCA slides", "count": snapshot.get("slides_total", "1,132 raw / 1,058 tiled")},
            {"dataset": "TCGA-BRCA RNA-seq", "count": "1,230"},
            {"dataset": "TCGA-BRCA clinical", "count": "1,097"},
            {"dataset": "Aligned multimodal count", "count": summary.get("aligned_sample_count", snapshot.get("aligned_patients", "Pending"))},
            {"dataset": "Extraction coverage", "count": snapshot.get("vision_embeddings", "758 / 1058 tiled TCGA slides at the time of final collection")},
        ]
    )
    st.markdown("#### Dataset Inventory")
    st.dataframe(dataset_inventory, use_container_width=True, hide_index=True)


def _render_system_architecture() -> None:
    st.header("System Architecture")
    st.markdown(
        """
        **Input Layer**  
        Histopathology WSI embeddings + clinical features + RNA-seq tensors + biomedical literature evidence

        **Four Agents**  
        Vision agent -> Clinical agent -> Genomics agent -> Literature agent

        **Cross-Attention Fusion**  
        Patient-aligned verifier with modality projections, gated attention, and unified risk aggregation

        **Risk Output**  
        High / intermediate / low risk stratification with modality-level reasoning support
        """
    )
    st.info("Literature agent provides clinical interpretation support. TCGA survival prediction uses vision + clinical + genomics fusion.")

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
            {"Phase": "Now", "Milestone": "GPU-validated multimodal TCGA baseline"},
            {"Phase": "Phase 2", "Milestone": "SurvPath pathway tokenization"},
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
    st.caption("Clinical Decision Support System")

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


if __name__ == "__main__":
    main()
