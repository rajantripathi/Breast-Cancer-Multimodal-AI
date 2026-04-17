# Breast Cancer Multimodal AI

Two-stage AI platform for breast cancer screening, diagnosis, and prognosis.

## Stage 1: Mammography Screening

- **Task:** Population-level breast cancer detection from mammograms
- **Data:** VinDr-Mammo (5,000 exams, 20,000 images)
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
- **Paper:** Submitted to Computer Methods and Programs in Biomedicine

## Mammography Status

- Default screening path: legacy VinDr-only ConvNeXt recipe
- Final mammography benchmark: `test_auroc = 0.7407`
- CMMD auxiliary ablation: `test_auroc = 0.7092`

## Foundation Models

- `UNI2` | Vision | `1536` dims | Active | Harvard/Mahmood Lab
- `CTransPath` | Vision | `768` dims | Active | Open access
- `Virchow` | Vision | `1280` dims | Approved | Paige AI
- `CONCH` | Vision-language | `512` dims | Pending | Harvard/Mahmood Lab

## Publications and References

- Pathomic Fusion
- PORPOISE
- SurvPath
- MMP
- UNI2
- TITAN
- MCAT
- SABCS 2025 ICM+ benchmark

## Infrastructure

- Isambard-AI national supercomputer (NVIDIA GH200, 32 GPUs)
- Feature extraction parallelised across GPU shards

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
