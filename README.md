# Breast Cancer Multimodal AI

Enterprise-ready clinical decision support system for breast cancer risk stratification, designed for Lenovo Healthcare Solutions, Intel Health & Life Sciences, and multidisciplinary oncology review. The platform fuses histopathology, clinical, genomics, and biomedical literature evidence into a unified patient risk assessment workflow.

## Quick Start

```bash
cd ~/Projects/Breast-Cancer-Multimodal-AI
streamlit run apps/streamlit_demo.py
```

For lightweight local validation:

```bash
python3 -m py_compile $(find . -name '*.py')
python3 -m orchestrator.run --all-sample-cases
```

## Results

- `559 / 1058` TCGA-BRCA slides have real UNI2 embeddings on Isambard
- `556` patient-aligned multimodal TCGA records are currently available for verifier training
- Latest verifier summary on Isambard reports:
  - `val_accuracy`: `1.0`
  - `num_train / num_val / num_test`: `389 / 83 / 84`
- Latest completed enterprise evaluation artifact reports:
  - `balanced_accuracy`: `1.0`
  - `f1_macro`: `1.0`
  - `ece`: `0.0`
  - `c_index`: pending, skipped when admissible survival pairs are insufficient

The current enterprise evaluation artifact is synchronized to the latest 556-patient TCGA alignment. See [`reports/tcga_results_summary.md`](/Users/rajantripathi/Projects/Breast-Cancer-Multimodal-AI/reports/tcga_results_summary.md) for the full boardroom summary, architecture notes, and limitations.

## Architecture

The system is built around four coordinated agents and a patient-level fusion verifier:

- Vision agent: UNI2-derived whole-slide pathology embeddings
- Clinical agent: structured risk features from TCGA clinical records
- Genomics agent: TCGA RNA-seq tensor analysis
- Literature agent: biomedical evidence retrieval layer for clinician context
- Cross-attention verifier: modality-specific projections, gated attention, and fused binary risk output

The Streamlit product demo provides:

- Patient Risk Assessment
- Multimodal Analysis
- Cohort Performance
- System Architecture

## Data

Primary development dataset: **TCGA-BRCA**

- `1,132` raw pathology slides
- `1,058` tiled slides prepared for feature extraction
- `1,230` RNA-seq records
- `1,097` clinical rows
- `556` currently aligned multimodal patient records

Planned Phase 2 expansion datasets include CPTAC-BRCA, METABRIC, CBIS-DDSM, VinDr-Mammo, and EMBED.

## Mammography Screening Layer (In Development)

The system is being extended with a mammography-based screening agent
that provides the front-end detection layer before the pathology
diagnostic pipeline. This creates a two-stage clinical AI platform:

1. **Screening** (mammography): Population-level breast cancer detection
2. **Diagnosis + Prognosis** (pathology + genomics + clinical): Individual patient risk assessment

### Architecture
- Input: 4-view digital mammogram (L-CC, R-CC, L-MLO, R-MLO)
- Encoder: ConvNeXt-Base (ImageNet pretrained, fine-tuned)
- Aggregation: Attention-weighted view fusion
- Output: Exam-level suspicion score

### Dataset
- VinDr-Mammo (PhysioNet): 5,000 exams with multi-reader BI-RADS annotations

### Status
- [x] Module structure created
- [x] Preprocessing pipeline
- [x] Model architecture
- [x] SLURM training scripts
- [x] VinDr-Mammo benchmark recovery
- [x] Model training
- [x] Evaluation and benchmarking
- [ ] Integration with pathology pipeline

Current screening benchmark:
- VinDr-only legacy recovery: `test_auroc = 0.7407`
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

- Training: Isambard-AI national supercomputer with NVIDIA GH200 Grace Hopper nodes
- Target inference platform: Lenovo ThinkSystem SR675i with Intel Xeon Scalable
- Edge deployment target: Lenovo ThinkEdge for hospital-local inference
- Optimization track: Intel OpenVINO for quantized deployment

## Team

- Dr Rajan Tripathi, Director AI2 Innovation Lab, NVIDIA DLI Ambassador
- AUT (ASU-powered), Tashkent
- Bikal Technologies, UK
- Research Associate, SOAS Centre for AI Futures, University of London

## Operational Notes

- Heavy preprocessing, TCGA extraction, training, and evaluation are designed to run on Isambard
- The local machine is used for editing, artifact inspection, demo delivery, and Git operations
- Do not commit real credentials; rotate any exposed tokens before production use
