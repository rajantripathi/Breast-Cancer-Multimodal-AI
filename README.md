# Breast Cancer Multimodal AI

> **Pathology foundation model benchmark for survival prediction.**
> CONCH V+C+G cross-attention architecture. C-index **0.609**. Stage 1 AUROC **0.741** (95% CI 0.649 to 0.825). Log-rank **p = 0.005**.

## Headline Results

| Metric | Value |
|---|---|
| Concordance Index (C-index) | 0.609 |
| Stage 1 AUROC | 0.741 (95% CI 0.649–0.825) |
| Median split log-rank | p = 0.005 |
| 5-year time-dependent AUC | 0.612 |

## Foundation Models Benchmarked

- **CONCH** (V+C+G cross-attention) — selected architecture
- **UNI2** — comparative baseline
- **CTransPath** — comparative baseline

Mammography module: ConvNeXt-Base baseline (AUROC 0.7407) retained as canonical; pathology pipeline used for survival prediction.

Stage 2 multimodal pathology benchmarking system for breast cancer
diagnosis and prognosis, with a two-stage screening-to-diagnosis
deployment path.

## Stage 2 Benchmark Contribution

- **Task:** Individual patient risk assessment from tissue + genomics + clinical
- **Data:** TCGA-BRCA (1,054 slides, 1,094 RNA-seq, 1,097 clinical)
- **Benchmark focus:** foundation encoder comparison under a shared multimodal survival pipeline
- **Vision encoders compared:** UNI2, CONCH, CTransPath
- **Best result:** CONCH V+C+G cross-attention, C-index `0.609 +/- 0.044`
- **Risk stratification:** Log-rank `p = 0.041` (`n = 1,043`)
- **Endpoint:** PFI per TCGA-CDR recommendation

## Encoder Benchmarking

- `CONCH` | Vision-language | `512` dims | Best downstream survival result
- `UNI2` | Vision | `1536` dims | Strong pathology baseline
- `CTransPath` | Vision | `768` dims | Open-access pathology encoder
- `Virchow` | Vision | `1280` dims | Approved for later comparison, not the current benchmark winner

## Clinical Deployment Context

The repository also carries the deployment path as a two-stage breast
cancer AI system:

1. **Stage 1: Mammography screening** for population-level detection
2. **Stage 2: Multimodal pathology diagnosis + prognosis** for patient-level risk assessment

### Stage 1: Mammography Screening

- **Task:** Population-level breast cancer detection from mammograms
- **Data:** VinDr-Mammo (5,000 exams, 20,000 images)
- **Auxiliary Data Path:** optional train-only public mammography expansion via metadata-compatible auxiliary datasets such as CBIS-DDSM
- **Model:** ConvNeXt-Base with 4-view attention fusion
- **Final mammography benchmark:** `test_auroc = 0.7407`
- **CMMD auxiliary ablation:** `test_auroc = 0.7092`
- **Module:** `agents/mammography/`

## Infrastructure

- Isambard-AI national supercomputer (NVIDIA GH200, 32 GPUs)
- Feature extraction parallelised across GPU shards

## Enterprise Architecture

This repository includes healthcare AI architecture material for governed clinical decision-support workflows:

- [`docs/architecture_blueprints.md`](docs/architecture_blueprints.md): implemented system, healthcare platform, and AWS managed-service diagrams
- [`docs/enterprise_architecture.md`](docs/enterprise_architecture.md): production reference architecture with guardrails, audit, monitoring, and operational controls
- [`docs/dialogxr_integration.md`](docs/dialogxr_integration.md): secure multimodal workflow mapping for a DialogXR-style platform
- [`docs/cv_project_summary.md`](docs/cv_project_summary.md): concise project summary and role-facing technical positioning

The project should be positioned as clinical decision support and research architecture, not autonomous diagnosis. The enterprise value is the combination of medical imaging, pathology foundation models, genomics, clinical features, statistical evaluation, Isambard workflows, and governance-aware deployment design.

## Scope

This repository is the project codebase for the two-stage breast cancer AI
workflow, including training, evaluation, orchestration, and paper-supporting
analysis artifacts. It is not tied to a single manuscript submission target.

## Repository Structure

```text
agents/
  mammography/          # Stage 1: screening
  vision/               # Stage 2: pathology vision encoder registry + extraction
  literature/           # Stage 2: PubMed RAG
training/               # Stage 2 multimodal fusion and survival training
evaluation/             # Metrics, significance utilities, and visualisation
orchestrator/           # Two-stage routing logic
apps/                   # Streamlit demo
slurm/                  # Isambard job scripts
reports/                # Results and paper artifacts
docs/paper/             # LaTeX manuscript
```
