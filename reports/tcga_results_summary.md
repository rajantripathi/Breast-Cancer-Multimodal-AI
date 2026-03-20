# TCGA Results Summary

## Executive Snapshot

- Evaluation status: latest completed evaluation artifact currently reflects the prior 173-patient aligned verifier run; refreshed 556-patient retrain and evaluation jobs are queued on Isambard as `3244425` and `3244426`
- Vision embeddings extracted: 559 / 1058 tiled TCGA slides
- Aligned patients: 556
- Train / validation / test: 389 / 83 / 84
- Architecture: UNI2 vision embeddings + TCGA genomics tensors + TCGA clinical features -> modality projections -> cross-attention verifier -> binary risk prediction

## Dataset Scale

- TCGA-BRCA slides: 1,132 raw slides, 1,058 tiled slides available for feature extraction
- TCGA-BRCA RNA-seq: 1,230 records
- TCGA-BRCA clinical rows: 1,097 rows
- Vision patients with embeddings: 558
- Genomics patients with tensors: 1,094
- Clinical patients with usable rows: 1,097
- Patient-aligned multimodal cohort: 556

## Current Model State

- Latest verifier summary on Isambard:
  - `val_accuracy`: `1.0`
  - `num_samples`: `556`
  - `num_train`: `389`
  - `num_val`: `83`
  - `num_test`: `84`
  - `alignment_status`: `patient_aligned_tcga`
  - `aligned_sample_count`: `556`
- Latest completed enterprise evaluation artifact:
  - `balanced_accuracy`: `1.0`
  - `f1_macro`: `1.0`
  - `ece`: `0.0`
  - `auroc_macro`: `0.0`
  - `auprc_macro`: `0.0`
  - `brier_score`: `0.0`
  - `auroc_ci_95`: `[0.0, 0.0]`
  - `balanced_accuracy_ci_95`: `[1.0, 1.0]`
  - `c_index_message`: `Survival labels present but no admissible pairs; C-index skipped`

## Interpretation

The cohort size and training summary are current and reflect the latest rebuilt TCGA crosswalk. The enterprise evaluation artifact is stale relative to the latest 556-patient retrain state and still reports the prior 173-patient alignment summary. The refreshed evaluation job has already been submitted and will replace these placeholder evaluation numbers once the current Isambard queue completes.

## Architecture Details

- Vision backbone: UNI2 pathology foundation model, 1536-dimensional slide embeddings
- Genomics input: TCGA RNA-seq tensors derived from the real genomics preprocessing pipeline
- Clinical input: normalized numeric features from `data/tcga_brca_clinical.csv`
- Fusion model: modality-specific projection layers, multi-head cross-attention, gated fusion, binary risk head
- Demo enhancement: per-modality risk predictions are now extracted from the trained verifier projection layers for clinical explanation in Streamlit

## Known Limitations

- Full TCGA slide extraction is still in progress; extraction coverage is not yet 100%
- The current enterprise evaluation artifact is not yet synchronized with the new 556-patient retrain
- C-index remains skipped when admissible survival pairs are insufficient
- Literature evidence is deployment-ready conceptually but not trained per patient, by design
- The demo currently relies on exported artifacts rather than live inference

## Phase 2 Targets

- Complete TCGA UNI2 extraction to full 1,058-slide coverage
- Refresh verifier evaluation on the 556-patient aligned cohort
- Add Cox survival loss for survival-optimized training
- Add SurvPath pathway tokenization for genomics
- Extend external validation to CPTAC-BRCA
- Profile inference on Lenovo and Intel-aligned deployment hardware
