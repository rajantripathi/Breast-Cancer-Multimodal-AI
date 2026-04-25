# Breast Cancer Multimodal AI

Two-stage AI platform for breast cancer screening, diagnosis, and prognosis.

## Stage 1: Mammography Screening

- **Task:** Population-level breast cancer detection from mammograms
- **Data:** VinDr-Mammo (5,000 exams, 20,000 images)
- **Auxiliary Data Path:** optional train-only public mammography expansion via metadata-compatible auxiliary datasets such as CBIS-DDSM
- **Model:** ConvNeXt-Base with 4-view attention fusion
- **Result:** Test AUROC 0.741
- **Module:** `agents/mammography/`

## Stage 2: Multimodal Pathology Diagnosis + Prognosis

- **Task:** Individual patient risk assessment from tissue + genomics + clinical
- **Data:** TCGA-BRCA (1,054 slides, 1,094 RNA-seq, 1,097 clinical)
- **Models:** UNI2, CONCH, CTransPath (foundation model comparison)
- **Best Result:** CONCH V+C+G cross-attention, C-index 0.609 +/- 0.044
- **Risk Stratification:** Log-rank p = 0.041 (n = 1,043)
- **Endpoint:** PFI per TCGA-CDR recommendation

## Infrastructure

- Isambard-AI national supercomputer (NVIDIA GH200, 32 GPUs)
- Feature extraction parallelised across GPU shards

## Scope

This repository is the project codebase for the two-stage breast cancer AI
workflow, including training, evaluation, orchestration, and paper-supporting
analysis artifacts. It is not tied to a single manuscript submission target.

## Repository Structure

```text
agents/
  mammography/          # Stage 1: screening
  vision/               # Stage 2: pathology vision encoder
  genomics/             # Stage 2: pathway genomics
  clinical/             # Stage 2: clinical features
  literature/           # Stage 2: PubMed RAG
training/               # Fusion training (Cox survival)
evaluation/             # Metrics and visualisation
orchestrator/           # Two-stage routing logic
apps/                   # Streamlit demo
slurm/                  # Isambard job scripts
reports/                # Results and paper artifacts
docs/paper/             # LaTeX manuscript
```
