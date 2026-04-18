# Breast Cancer Multimodal AI

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
- **Paper:** Submitted to Computer Methods and Programs in Biomedicine

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

- Default screening path: legacy VinDr-only ConvNeXt recipe
- Final mammography benchmark: `test_auroc = 0.7407`
- CMMD auxiliary ablation: `test_auroc = 0.7092`
- Module: `agents/mammography/`

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
