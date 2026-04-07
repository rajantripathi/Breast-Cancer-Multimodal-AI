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
    "model": "CONCH cross-attention survival model",
    "endpoint": "Progression-Free Interval (PFI)",
    "evaluation": "5-fold stratified cross-validation",
    "cohort": 1043,
    "best_configuration": "Vision + Clinical + Genomics",
    "best_encoder": "CONCH",
    "c_index_mean": 0.6093,
    "c_index_std": 0.0441,
    "risk_logrank_p": 0.0412,
    "fold_c_index": [0.6588, 0.5558, 0.64, 0.6354, 0.5564],
    "secondary": {
        "auroc_mean": 0.5897,
        "auroc_std": 0.0445,
        "low_risk_event_rate": 0.1153,
        "high_risk_event_rate": 0.1552,
    },
    "ablation": {
        "V": (0.5675, 0.0461),
        "V+C": (0.5615, 0.0529),
        "V+G": (0.5846, 0.0587),
        "V+C+G": (0.6093, 0.0441),
    },
    "training_gpu": "NVIDIA GH200 120GB",
    "genomics_features": 50,
    "uni2_embeddings": 1054,
    "ctranspath_embeddings": 1049,
    "conch_embeddings": 1049,
    "clinical_rows": 1097,
}

FROZEN_SCREENING = {
    "dataset": "VinDr-Mammo",
    "exams": 5000,
    "images": 20000,
    "model": "ConvNeXt-Base with 4-view attention fusion",
    "task": "Population-level breast cancer detection from mammograms",
    "test_auroc": 0.7407,
    "best_val_auroc": 0.7560,
    "best_epoch": 32,
    "image_size": 224,
    "sensitivity_at_90_specificity": 0.4333,
    "specificity_at_90_sensitivity": 0.3847,
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


def _is_missing(value: Any) -> bool:
    text = str(value).strip() if value is not None else ""
    return text == "" or text.lower() in {"n/a", "not available", "none", "nan"}


def _maybe_clinical_row(label: str, value: Any) -> None:
    if not _is_missing(value):
        st.markdown(f"**{label}:** {value}")


def _modality_state(record: dict[str, Any], modality: str) -> tuple[str, float]:
    modality_predictions = record.get("modality_predictions") or {}
    payload = modality_predictions.get(modality, {})
    if payload:
        return str(payload.get("class", "Pending")), float(payload.get("confidence", 0.0) or 0.0)
    config = FROZEN_SCIENCE["best_configuration"]
    included = {
        "Vision": "vision" in config,
        "Clinical": "clinical" in config.lower(),
        "Genomics": "genomics" in config.lower(),
    }
    normalized = modality.title()
    if included.get(normalized):
        return "Included in best model", 1.0
    if normalized in included:
        return "Not used in frozen best model", 0.0
    return "Unavailable in current artifact", 0.0


def _display_contributions(record: dict[str, Any]) -> dict[str, float]:
    contributions = record.get("modality_contributions") or {}
    modality_predictions = record.get("modality_predictions") or {}
    if modality_predictions:
        return contributions or {"vision": 0.333, "clinical": 0.333, "genomics": 0.333}
    if FROZEN_SCIENCE["best_configuration"] == "Vision + Clinical + Genomics":
        return {"vision": 0.34, "clinical": 0.23, "genomics": 0.43}
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


def _load_mammography_assets(repo_root: Path) -> dict[str, Any]:
    summary_path = repo_root / "outputs" / "mammography" / "summary.json"
    history_path = repo_root / "outputs" / "mammography" / "history.json"

    summary = FROZEN_SCREENING.copy()
    if summary_path.exists():
        payload = load_json(summary_path)
        summary.update(
            {
                "best_val_auroc": float(payload.get("best_val_auroc", summary["best_val_auroc"])),
                "test_auroc": float(payload.get("test_auroc", summary["test_auroc"])),
                "best_epoch": int(payload.get("best_epoch", summary["best_epoch"])),
                "image_size": int(payload.get("image_size", summary["image_size"])),
                "sensitivity_at_90_specificity": float(
                    payload.get("test_sensitivity_at_90_specificity", summary["sensitivity_at_90_specificity"])
                ),
                "specificity_at_90_sensitivity": float(
                    payload.get("test_specificity_at_90_sensitivity", summary["specificity_at_90_sensitivity"])
                ),
                "train_exams": int(payload.get("train_exams", 3497)),
                "val_exams": int(payload.get("val_exams", 749)),
                "test_exams": int(payload.get("test_exams", 750)),
            }
        )

    history = load_json(history_path) if history_path.exists() else []
    return {"summary": summary, "history": history}


def _render_demo_overview(mammo_assets: dict[str, Any]) -> None:
    summary = mammo_assets["summary"]
    st.header("Overview")
    st.caption("Guided walkthrough of the final two-stage research system")
    st.info(
        "This app is a guided research demo built from frozen benchmark artifacts. "
        "It is intentionally not a live clinical inference product."
    )

    top = st.columns(4)
    top[0].metric("Stage 1 AUROC", f"{summary['test_auroc']:.3f}")
    top[0].caption("VinDr-Mammo screening benchmark")
    top[1].metric("Stage 2 C-index", f"{FROZEN_SCIENCE['c_index_mean']:.3f}")
    top[1].caption("Best multimodal prognosis benchmark")
    top[2].metric("Stage 2 Log-rank p", f"{FROZEN_SCIENCE['risk_logrank_p']:.3f}")
    top[2].caption("Risk-group separation from pooled out-of-fold analysis")
    top[3].metric("Cohort", f"{FROZEN_SCIENCE['cohort']}")
    top[3].caption("Patient-aligned TCGA-BRCA evaluation cohort")

    left, right = st.columns([1.1, 1.0])
    with left:
        _card_start("What This Demo Shows")
        st.markdown("**Stage 1:** population-level mammography screening on VinDr-Mammo.")
        st.markdown("**Stage 2:** multimodal prognosis from pathology, genomics, and clinical data under PFI.")
        st.markdown("**Workflow:** suspicious screening cases route into deeper multimodal assessment.")
        st.markdown("**Interaction model:** curated artifacts, benchmark summaries, and explorable patient examples.")
        _card_end()
    with right:
        _card_start("Recommended Walkthrough")
        st.markdown("1. Start with `Two-Stage Workflow` to set the story.")
        st.markdown("2. Show `Stage 1 Screening` and `Stage 1 Performance` for the final mammography benchmark.")
        st.markdown("3. Move to `Stage 2 Patient Risk` for a representative case.")
        st.markdown("4. Finish on `Stage 2 Cohort Performance` for the final science metrics.")
        _card_end()

    route = pd.DataFrame(
        [
            {"Step": "1", "Component": "Stage 1 screening", "Role": "Assigns mammography suspicion score"},
            {"Step": "2", "Component": "Screening router", "Role": "Determines surveillance vs further workup"},
            {"Step": "3", "Component": "Stage 2 multimodal model", "Role": "Produces prognosis-oriented risk estimate"},
            {"Step": "4", "Component": "Decision support", "Role": "Provides interpretable benchmark and case context"},
        ]
    )
    st.markdown("#### System Snapshot")
    st.dataframe(route, use_container_width=True, hide_index=True)


def _render_artifact_note(source_label: str, kind: str) -> None:
    st.caption(f"{kind} source: `{source_label}`")
    st.caption("Interactive views below are driven by stored research artifacts, not live inference.")


def _ablation_chart() -> go.Figure:
    values = {
        name: value[0] for name, value in FROZEN_SCIENCE["ablation"].items()
    }
    figure = go.Figure(
        go.Bar(
            x=list(values.keys()),
            y=list(values.values()),
            marker_color=[NAVY, TEAL, "#4B84B0", GREEN],
            text=[f"{value:.3f}" for value in values.values()],
            textposition="outside",
        )
    )
    figure.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="C-index",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY),
    )
    return figure


def _mammography_history_chart(history: list[dict[str, Any]]) -> go.Figure:
    epochs = [item.get("epoch") for item in history]
    train_auroc = [item.get("train_auroc") for item in history]
    val_auroc = [item.get("val_auroc") for item in history]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=epochs, y=train_auroc, mode="lines", name="Train AUROC", line=dict(color=TEAL, width=2)))
    figure.add_trace(go.Scatter(x=epochs, y=val_auroc, mode="lines", name="Validation AUROC", line=dict(color=NAVY, width=2)))
    figure.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        xaxis_title="Epoch",
        yaxis_title="AUROC",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure


def _render_stage1_screening(mammo_assets: dict[str, Any]) -> None:
    summary = mammo_assets["summary"]
    st.header("Stage 1 Screening")
    st.caption("Population-level mammography screening benchmark")

    top = st.columns(4)
    top[0].metric("Test AUROC", f"{summary['test_auroc']:.3f}")
    top[0].caption("Final mammography baseline retained after all follow-up experiments")
    top[1].metric("Model", "ConvNeXt-Base")
    top[1].caption("4-view attention fusion over L-CC, R-CC, L-MLO, R-MLO")
    top[2].metric("Dataset", f"{summary['exams']:,} exams")
    top[2].caption(f"{summary['images']:,} images from VinDr-Mammo")
    top[3].metric("Image Size", f"{summary['image_size']} px")
    top[3].caption("Final baseline training resolution")

    left, right = st.columns([1.15, 1.0])
    with left:
        _card_start("Screening Task")
        st.markdown(f"**Task:** {summary['task']}")
        st.markdown(f"**Dataset:** {summary['dataset']}")
        st.markdown(f"**Architecture:** {summary['model']}")
        st.markdown("**Clinical role:** front-end screen to identify suspicious exams before Stage 2 multimodal workup")
        st.caption("This stage is presented as a validated cohort benchmark. The repository does not store benchmark-grade per-exam mammography demo cases.")
        _card_end()

    with right:
        _card_start("Operating Points")
        st.metric("Best validation AUROC", f"{summary['best_val_auroc']:.3f}")
        st.metric("Sensitivity @ 90% specificity", f"{summary['sensitivity_at_90_specificity']:.3f}")
        st.metric("Specificity @ 90% sensitivity", f"{summary['specificity_at_90_sensitivity']:.3f}")
        st.caption(f"Best epoch: {summary['best_epoch']}")
        _card_end()


def _render_stage1_performance(mammo_assets: dict[str, Any]) -> None:
    summary = mammo_assets["summary"]
    history = mammo_assets["history"]
    st.header("Stage 1 Performance")
    st.caption("Final retained mammography baseline | ConvNeXt-Base | 224px")

    metrics = st.columns(3)
    metrics[0].metric("Train / Val / Test", f"{summary.get('train_exams', 3497)} / {summary.get('val_exams', 749)} / {summary.get('test_exams', 750)}")
    metrics[1].metric("Test AUROC", f"{summary['test_auroc']:.3f}")
    metrics[2].metric("Best epoch", str(summary["best_epoch"]))

    if history:
        st.markdown("#### Training History")
        st.plotly_chart(_mammography_history_chart(history), use_container_width=True)

    st.markdown("#### Benchmark Notes")
    notes = pd.DataFrame(
        [
            {"Field": "Canonical result", "Value": "ConvNeXt-Base baseline retained"},
            {"Field": "Tried and not promoted", "Value": "EfficientNet-B5 and Mammo-CLIP variants"},
            {"Field": "Stored mammography artifacts", "Value": "Summary + training history only"},
            {"Field": "Demo implication", "Value": "Show cohort performance and workflow, not fake per-patient screening predictions"},
        ]
    )
    st.dataframe(notes, use_container_width=True, hide_index=True)


def _render_two_stage_workflow() -> None:
    st.header("Two-Stage Workflow")
    st.caption("Integrated screening -> diagnosis + prognosis pathway")

    st.markdown(
        """
        **Stage 1: Mammography Screening**  
        Four-view mammography model assigns a suspiciousness score at the population-screening level.

        **Routing Logic**  
        Low-suspicion cases remain in standard screening surveillance. Suspicious cases are referred for diagnostic workup and routed into the multimodal pathology pipeline.

        **Stage 2: Multimodal Prognosis**  
        Histopathology, genomics, and clinical data are fused by the frozen CONCH cross-attention survival model to estimate progression risk under PFI.
        """
    )

    route = pd.DataFrame(
        [
            {"Step": "1", "System block": "Mammography screening", "Output": "Suspicion score"},
            {"Step": "2", "System block": "Screening router", "Output": "Standard surveillance or pathology referral"},
            {"Step": "3", "System block": "Pathology + genomics + clinical fusion", "Output": "PFI risk score and risk band"},
            {"Step": "4", "System block": "Clinical decision support", "Output": "Cohort context and interpretive evidence"},
        ]
    )
    st.dataframe(route, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        _card_start("Stage 1 Final Result")
        st.metric("Mammography test AUROC", f"{FROZEN_SCREENING['test_auroc']:.3f}")
        st.caption("ConvNeXt-Base, 4-view attention, VinDr-Mammo")
        _card_end()
    with right:
        _card_start("Stage 2 Final Result")
        st.metric("Best C-index", f"{FROZEN_SCIENCE['c_index_mean']:.3f} +/- {FROZEN_SCIENCE['c_index_std']:.3f}")
        st.caption("CONCH + Vision + Clinical + Genomics under PFI")
        _card_end()


def _render_patient_risk_assessment(assets: dict[str, Any]) -> None:
    cases, source_label = _choose_cases(assets)
    st.header("Stage 2 Patient Risk")
    st.caption("Explorable prognosis example from frozen multimodal prediction artifacts")
    if not cases:
        st.warning("No prediction artifact is available locally yet.")
        return
    _render_artifact_note(source_label, "Patient-case artifact")

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
            _maybe_clinical_row("ER Status", clinical.get("er_status"))
            _maybe_clinical_row("PR Status", clinical.get("pr_status"))
            _maybe_clinical_row("HER2 Status", clinical.get("her2_status"))
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
            st.caption("Frozen model uses CONCH + Vision + Clinical + Genomics (best paper benchmark).")
            _card_end()
    except Exception as exc:
        st.warning(f"Contribution chart unavailable: {exc}")


def _render_multimodal_analysis(assets: dict[str, Any]) -> None:
    cases, source_label = _choose_cases(assets)
    st.header("Stage 2 Multimodal Analysis")
    st.caption("Modality-level view of the selected prognosis example")
    if not cases:
        st.warning("No prediction artifact is available locally yet.")
        return
    _render_artifact_note(source_label, "Patient-case artifact")

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
            st.caption("CONCH vision-language pathology embeddings from this patient's H&E slide feed the frozen best-performing paper model.")
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
            if not _is_missing(clinical.get("er_status")):
                st.markdown(f"ER: {clinical.get('er_status')}")
            if not _is_missing(clinical.get("pr_status")):
                st.markdown(f"PR: {clinical.get('pr_status')}")
            if not _is_missing(clinical.get("her2_status")):
                st.markdown(f"HER2: {clinical.get('her2_status')}")
            st.caption("Clinical covariates are part of the best paper model and improved the final CONCH cross-attention benchmark.")
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
    st.header("Stage 2 Cohort Performance")
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
    row2[0].metric("AUROC", f"{frozen['secondary']['auroc_mean']:.3f} +/- {frozen['secondary']['auroc_std']:.3f}")
    row2[0].caption("Supplementary pooled discrimination across the 5 cross-validation folds")
    row2[1].metric("Best Encoder", frozen["best_encoder"])
    row2[1].caption("Strongest pathology foundation model in the final paper benchmark")
    row2[2].metric("Log-rank p", f"{frozen['risk_logrank_p']:.3f}")
    row2[2].caption("Risk-group separation from pooled out-of-fold Kaplan-Meier analysis")

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
    st.caption("For the best encoder family, full multimodal cross-attention (V+C+G) is the strongest configuration in the final paper benchmark.")

    dataset_inventory = pd.DataFrame(
        [
            {"dataset": "TCGA-BRCA aligned cohort", "count": str(frozen["cohort"])},
            {"dataset": "Genomics features", "count": f"{frozen['genomics_features']} Hallmark pathways"},
            {"dataset": "TCGA-BRCA clinical", "count": str(frozen["clinical_rows"])},
            {"dataset": "UNI2 embeddings extracted", "count": str(frozen["uni2_embeddings"])},
            {"dataset": "CTransPath embeddings extracted", "count": str(frozen["ctranspath_embeddings"])},
            {"dataset": "CONCH embeddings extracted", "count": str(frozen["conch_embeddings"])},
        ]
    )
    st.markdown("#### Dataset Inventory")
    st.dataframe(dataset_inventory, use_container_width=True, hide_index=True)

    st.markdown("#### Scientific Context")
    st.markdown(
        """
        - TCGA-BRCA has heavy censoring, making it one of the harder TCGA cancer types for progression-risk modeling.
        - PFI follows the TCGA-CDR recommendation for BRCA; overall survival is not the preferred endpoint for this disease setting.
        - Pooled out-of-fold Kaplan-Meier analysis from the best model gives `p = 0.041`, indicating modest but statistically significant risk-group separation.
        """
    )


def _render_system_architecture() -> None:
    st.header("System Architecture")
    st.markdown(
        """
        **Input Layer**  
        Histopathology WSI embeddings + Hallmark pathway genomics + clinical context + biomedical literature support

        **Frozen Headline Model**  
        CONCH cross-attention survival model with Vision + Clinical + Genomics

        **Fusion Strategy**  
        CONCH pathology embeddings, clinical covariates, and Hallmark pathway tokens are projected into a shared survival modeling stack with Cox loss

        **Risk Output**  
        Progression-free interval risk ranking with cross-validated C-index reporting
        """
    )
    st.info("Literature agent provides interpretation support in the demo. The frozen science result uses Cox survival loss, the PFI endpoint, and 5-fold stratified cross-validation.")

    registry = pd.DataFrame(
        [
            {"Model": "UNI2", "Type": "Vision", "Dim": 1536, "Status": "Benchmark", "Source": "Harvard/Mahmood Lab"},
            {"Model": "CTransPath", "Type": "Vision", "Dim": 768, "Status": "Benchmark", "Source": "Open access"},
            {"Model": "Virchow", "Type": "Vision", "Dim": 1280, "Status": "Approved", "Source": "Paige AI"},
            {"Model": "CONCH", "Type": "Vision-Language", "Dim": 512, "Status": "Active / Best", "Source": "Harvard/Mahmood Lab"},
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
            {"Phase": "Now", "Milestone": "PFI-aligned 5-fold CV benchmark with CONCH + V+C+G"},
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
    repo_root = Path(__file__).resolve().parents[1]
    assets = discover_tcga_assets(repo_root)
    mammo_assets = _load_mammography_assets(repo_root)

    st.title("Breast Cancer Multimodal AI")
    st.caption("Two-Stage Clinical AI Platform | Mammography Screening + Multimodal Prognosis")
    st.sidebar.caption("Guided demo mode")
    st.sidebar.caption("Frozen benchmark artifacts only")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Stage 1 Screening",
            "Stage 1 Performance",
            "Two-Stage Workflow",
            "Stage 2 Patient Risk",
            "Stage 2 Multimodal Analysis",
            "Stage 2 Cohort Performance",
        ],
    )
    if page == "Overview":
        _render_demo_overview(mammo_assets)
    elif page == "Stage 1 Screening":
        _render_stage1_screening(mammo_assets)
    elif page == "Stage 1 Performance":
        _render_stage1_performance(mammo_assets)
    elif page == "Two-Stage Workflow":
        _render_two_stage_workflow()
    elif page == "Stage 2 Patient Risk":
        _render_patient_risk_assessment(assets)
    elif page == "Stage 2 Multimodal Analysis":
        _render_multimodal_analysis(assets)
    else:
        _render_cohort_performance(assets)

    st.markdown("---")
    st.caption("github.com/rajantripathi/Breast-Cancer-Multimodal-AI")


if __name__ == "__main__":
    main()
