from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from apps.utils import discover_tcga_assets

PRIMARY_BLUE = "#1B3A5C"
SOFT_GREY = "#EFF3F7"
GREEN = "#2E8B57"
AMBER = "#D99100"
RED = "#C0392B"


def _choose_prediction_source(assets: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if assets["verifier_predictions"]:
        return assets["verifier_predictions"], "outputs/tcga_verifier/predictions.json"
    legacy_predictions = assets["paths"]["verifier_predictions"].parent.parent / "verifier" / "predictions.json"
    if legacy_predictions.exists():
        from apps.utils import load_json

        return load_json(legacy_predictions), "outputs/verifier/predictions.json"
    if assets["sample_case_results"]:
        return assets["sample_case_results"], "outputs/sample_case_results.json"
    return [], "No prediction artifact available"


def _risk_band(score: float) -> tuple[str, str]:
    if score >= 0.67:
        return "HIGH RISK", RED
    if score >= 0.33:
        return "INTERMEDIATE", AMBER
    return "LOW RISK", GREEN


def _recommendation(label: str) -> str:
    if label == "HIGH RISK":
        return "Recommend oncology team review. Consider neoadjuvant therapy evaluation."
    if label == "INTERMEDIATE":
        return "Recommend follow-up imaging in 3 months. Consider genomic testing if not yet performed."
    return "Standard surveillance protocol. Annual mammography recommended."


def _extract_score(record: dict[str, Any]) -> float:
    if "risk_score" in record:
        return float(record["risk_score"])
    probabilities = record.get("probabilities", {})
    return float(probabilities.get("high_concern", 0.0))


def _extract_confidence(record: dict[str, Any]) -> float:
    if "confidence" in record:
        return float(record["confidence"])
    probabilities = record.get("probabilities", {})
    if probabilities:
        return float(max(probabilities.values()))
    return 0.0


def _profile_rows(record: dict[str, Any]) -> list[tuple[str, Any]]:
    rows = [
        ("Patient ID", record.get("sample_id", record.get("patient_id", "Pending"))),
        ("Ground Truth", record.get("true_label", "Pending")),
        ("Predicted Label", record.get("predicted_label", record.get("fused_label", "Pending"))),
        ("Alignment", record.get("alignment_status", "Pending")),
    ]
    if "survival_time" in record:
        rows.append(("Survival Time", record.get("survival_time")))
    if "event_observed" in record:
        rows.append(("Event Observed", record.get("event_observed")))
    return rows


def _modality_contribution_chart(record: dict[str, Any]) -> tuple[go.Figure, str | None]:
    contributions = record.get("modality_contributions")
    note = None
    if not contributions:
        contributions = {"vision": 0.3333, "clinical": 0.3333, "genomics": 0.3334}
        note = "Per-modality contribution weights are unavailable in this artifact. Showing equal weights as a placeholder."
    figure = go.Figure(
        data=[
            go.Pie(
                labels=[label.title() for label in contributions],
                values=list(contributions.values()),
                hole=0.58,
                marker=dict(colors=[PRIMARY_BLUE, "#4E79A7", "#9DB7D5"]),
                textinfo="label+percent",
            )
        ]
    )
    figure.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PRIMARY_BLUE),
        showlegend=False,
    )
    return figure, note


def _modality_panel(record: dict[str, Any], key: str, title: str, icon: str, fallback_text: str) -> None:
    payload = (record.get("modality_predictions") or {}).get(key, {})
    predicted_class = payload.get("class", "Pending")
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    st.markdown(f"### {icon} {title}")
    st.markdown(f"**Prediction:** {predicted_class}")
    st.progress(min(max(confidence, 0.0), 1.0))
    if key == "literature" and not payload:
        st.caption("Literature agent available in full deployment")
    else:
        st.caption(fallback_text)


def _metric_text(value: Any) -> str:
    if value is None:
        return "Pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _inject_style() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(27, 58, 92, 0.13), transparent 28%),
                linear-gradient(180deg, #ffffff 0%, {SOFT_GREY} 100%);
        }}
        .block-container {{
            max-width: 1260px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }}
        .hero-card {{
            background: white;
            border: 1px solid rgba(27,58,92,0.12);
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 18px 38px rgba(27,58,92,0.08);
        }}
        .risk-badge {{
            display: inline-block;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: white;
        }}
        h1, h2, h3 {{
            color: {PRIMARY_BLUE};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_patient_risk_assessment(assets: dict[str, Any]) -> None:
    predictions, source_label = _choose_prediction_source(assets)
    st.header("Patient Risk Assessment")
    st.caption(f"Prediction source: {source_label}")
    if not predictions:
        st.warning("No prediction artifact is available locally yet. This page will populate automatically when verifier predictions are generated.")
        return
    options = [item.get("sample_id", item.get("patient_id", f"record_{idx}")) for idx, item in enumerate(predictions)]
    selected_id = st.selectbox("Patient selector", options=options)
    record = next(item for item in predictions if item.get("sample_id", item.get("patient_id")) == selected_id)
    risk_score = _extract_score(record)
    confidence = _extract_confidence(record)
    risk_label, color = _risk_band(risk_score)
    chart, note = _modality_contribution_chart(record)

    left, centre, right = st.columns([1.1, 1.1, 1.2])
    with left:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("#### Patient Profile")
        for label, value in _profile_rows(record):
            st.markdown(f"**{label}:** {value}")
        st.markdown("</div>", unsafe_allow_html=True)
    with centre:
        st.markdown('<div class="hero-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("#### Risk Indicator")
        st.markdown(
            f'<div class="risk-badge" style="background:{color};">{risk_label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {risk_score:.4f}")
        st.caption(f"Confidence: {confidence:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("#### Modality Contribution")
        st.plotly_chart(chart, use_container_width=True)
        if note:
            st.caption(note)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Clinical Recommendation")
    st.info(_recommendation(risk_label))


def _render_multimodal_analysis(assets: dict[str, Any]) -> None:
    predictions, source_label = _choose_prediction_source(assets)
    st.header("Multimodal Analysis")
    st.caption(f"Prediction source: {source_label}")
    if not predictions:
        st.warning("No prediction artifact is available locally yet. This page will populate automatically when verifier predictions are generated.")
        return
    options = [item.get("sample_id", item.get("patient_id", f"record_{idx}")) for idx, item in enumerate(predictions)]
    selected_id = st.selectbox("Patient selector", options=options, key="analysis_patient")
    record = next(item for item in predictions if item.get("sample_id", item.get("patient_id")) == selected_id)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _modality_panel(
            record,
            "vision",
            "Vision Agent",
            "🔬",
            "Histopathology assessment based on UNI2 foundation model analysis of whole-slide tissue morphology",
        )
    with col2:
        _modality_panel(
            record,
            "clinical",
            "Clinical Agent",
            "📋",
            "Clinical risk factors including age, stage, receptor status",
        )
    with col3:
        _modality_panel(
            record,
            "genomics",
            "Genomics Agent",
            "🧬",
            "Molecular profile analysis using PAM50 gene panel and pathway scoring",
        )
    with col4:
        _modality_panel(
            record,
            "literature",
            "Literature Agent",
            "📚",
            "Evidence retrieval from biomedical literature (PubMed)",
        )


def _render_cohort_performance(assets: dict[str, Any]) -> None:
    metrics = assets["enterprise_metrics"] or {}
    summary = assets["verifier_summary"] or {}
    snapshot = assets["results_snapshot"] or {}
    st.header("Cohort Performance")
    st.caption("Enterprise Evaluation Metrics")
    row1 = st.columns(3)
    row1[0].metric("AUROC", _metric_text(metrics.get("auroc_macro")))
    row1[1].metric("F1", _metric_text(metrics.get("f1_macro")))
    row1[2].metric("Balanced Accuracy", _metric_text(metrics.get("balanced_accuracy")))

    row2 = st.columns(3)
    row2[0].metric("Brier Score", _metric_text(metrics.get("brier_score")))
    row2[1].metric("ECE", _metric_text(metrics.get("ece")))
    row2[2].metric("C-index", _metric_text(metrics.get("c_index") or metrics.get("c_index_message")))

    training_details = pd.DataFrame(
        [
            {"field": "Aligned patients", "value": summary.get("aligned_sample_count", snapshot.get("aligned_patients", "Pending"))},
            {"field": "Train / val / test split", "value": snapshot.get("train_split", "Pending")},
            {"field": "Epochs", "value": "100"},
            {"field": "Architecture", "value": "Three modality projections + multi-head cross-attention fusion + binary risk head"},
        ]
    )
    st.markdown("#### Training Details")
    st.dataframe(training_details, use_container_width=True, hide_index=True)

    dataset_inventory = pd.DataFrame(
        [
            {"dataset": "TCGA-BRCA slides", "count": snapshot.get("slides_total", "1,132 raw / 1,058 tiled")},
            {"dataset": "TCGA-BRCA RNA-seq", "count": snapshot.get("rnaseq_total", "1,230")},
            {"dataset": "TCGA-BRCA clinical", "count": snapshot.get("clinical_total", "1,097")},
            {"dataset": "Aligned multimodal count", "count": summary.get("aligned_sample_count", snapshot.get("aligned_patients", "Pending"))},
            {"dataset": "Extraction coverage", "count": snapshot.get("vision_embeddings", "Pending")},
        ]
    )
    st.markdown("#### Dataset Inventory")
    st.dataframe(dataset_inventory, use_container_width=True, hide_index=True)
    st.info(
        f"Validated on {summary.get('aligned_sample_count', snapshot.get('aligned_patients', 'N'))} patient-aligned multimodal records from TCGA-BRCA, "
        "the largest public breast cancer dataset with matched histopathology, genomics, and clinical outcomes."
    )


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

    model_registry = pd.DataFrame(
        [
            {"Model": "UNI2", "Type": "Vision", "Dim": 1536, "Status": "Active", "Source": "Harvard/Mahmood Lab"},
            {"Model": "CTransPath", "Type": "Vision", "Dim": 768, "Status": "Active", "Source": "Open access"},
            {"Model": "Virchow", "Type": "Vision", "Dim": 1280, "Status": "Approved", "Source": "Paige AI"},
            {"Model": "CONCH", "Type": "Vision-Language", "Dim": 512, "Status": "Pending", "Source": "Harvard/Mahmood Lab"},
        ]
    )
    st.markdown("#### Foundation Model Registry")
    st.dataframe(model_registry, use_container_width=True, hide_index=True)

    st.markdown("#### Infrastructure Alignment")
    st.markdown(
        """
        - Training: Isambard-AI national supercomputer (NVIDIA GH200 Grace Hopper)
        - Inference target: Lenovo ThinkSystem SR675i + Intel Xeon Scalable
        - Edge deployment: Lenovo ThinkEdge for hospital-local inference
        - Optimization: Intel OpenVINO for INT8 quantized inference
        """
    )
    st.markdown("#### Research Foundation")
    st.markdown(
        """
        - Pathomic Fusion
        - PORPOISE
        - SurvPath
        - MMP
        - UNI2
        - TITAN
        """
    )
    st.markdown("#### Phase 2 Roadmap")
    st.markdown(
        """
        - Cox survival loss optimization
        - SurvPath pathway tokenization
        - Multi-site external validation (CPTAC-BRCA)
        - Federated learning for privacy-preserving multi-hospital training
        - Full TCGA extraction on Lenovo ThinkSystem infrastructure
        - OpenVINO inference profiling on Intel Xeon
        """
    )

    datasets = pd.DataFrame(
        [
            {"Dataset": "CPTAC-BRCA", "Patients": 122, "Modalities": "WSI+proteomics+genomics", "Purpose": "External validation"},
            {"Dataset": "METABRIC", "Patients": 2509, "Modalities": "Expression+clinical+CNV", "Purpose": "Genomics benchmark"},
            {"Dataset": "CBIS-DDSM", "Patients": 2620, "Modalities": "Mammography", "Purpose": "Radiology modality"},
            {"Dataset": "VinDr-Mammo", "Patients": 5000, "Modalities": "Mammography", "Purpose": "Multi-reader benchmark"},
            {"Dataset": "EMBED", "Patients": "116,000+", "Modalities": "Mammography+clinical", "Purpose": "Scale validation"},
        ]
    )
    st.markdown("#### Available Public Datasets for Phase 2 Expansion")
    st.dataframe(datasets, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Breast Cancer Multimodal AI",
        page_icon="BC",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()
    repo_root = Path(__file__).resolve().parents[1]
    assets = discover_tcga_assets(repo_root)

    st.title("Breast Cancer Multimodal AI")
    st.caption("Clinical Decision Support System")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Patient Risk Assessment",
            "Multimodal Analysis",
            "Cohort Performance",
            "System Architecture",
        ],
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
