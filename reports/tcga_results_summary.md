# TCGA Results Summary

## Executive Snapshot

- Evaluation status: Cox-aware evaluation is fixed, but the current `689`-patient aligned cohort has no populated event labels, so proposal-grade held-out survival metrics are still blocked by the clinical endpoint data
- Vision embeddings extracted: 695 / 1058 tiled TCGA slides
- Aligned patients: 689
- Train / validation / test: currently blocked for the corrected cohort because the aligned endpoint distribution is single-class
- Architecture: UNI2 vision embeddings + TCGA genomics tensors + TCGA clinical features -> modality projections -> cross-attention verifier -> binary risk prediction

## Dataset Scale

- TCGA-BRCA slides: 1,132 raw slides, 1,058 tiled slides available for feature extraction
- TCGA-BRCA RNA-seq: 1,230 records
- TCGA-BRCA clinical rows: 1,097 rows
- Vision patients with embeddings: 693
- Genomics patients with tensors: 1,094
- Clinical patients with usable rows: 1,097
- Patient-aligned multimodal cohort: 689

## Current Model State

- Latest verifier summary on Isambard:
  - previous completed full-model run on the corrected pass is unavailable because the verifier job now fails fast when CUDA is not available
  - most recent attempted corrected run:
    - aligned cohort: `689`
    - full model job: failed early with `AssertionError: CUDA not available`
- Latest enterprise evaluation artifact:
  - `num_predictions`: `103`
  - `alignment_summary`: `Verifier trained on 681 patient-aligned bundles`
  - `balanced_accuracy`: `0.5631`
  - `f1_macro`: `0.3602`
  - `ece`: `0.0423`
  - `auroc_macro`: `0.0`
  - `auprc_macro`: `0.0`
  - `brier_score`: `0.2489`
  - `auroc_ci_95`: `[0.0, 0.0]`
  - `balanced_accuracy_ci_95`: `[0.466, 0.6602]`
  - `fused_label_distribution`: `{"high_concern": 45, "monitor": 58}`
  - `c_index_message`: `Survival labels present but no admissible pairs; C-index skipped`
  - `survival_time_diagnostic`: `min=0.0, max=7777.0, unique=87`
  - `event_diagnostic`: `sum=0, total=103`

## Interpretation

The Cox-aware evaluator is now correctly reading `risk_score`, `predicted_label`, and survival diagnostics from the verifier artifacts. The current blocker is not the evaluator anymore. It is the aligned clinical endpoint data: the latest `689`-patient aligned cohort has blank `vital_status` and blank `days_to_death` for all matched rows, so neither stratified splits nor held-out survival evaluation can produce meaningful final metrics on that refreshed cohort.

## Architecture Details

- Vision backbone: UNI2 pathology foundation model, 1536-dimensional slide embeddings
- Genomics input: TCGA RNA-seq tensors derived from the real genomics preprocessing pipeline
- Clinical input: normalized numeric features from `data/tcga_brca_clinical.csv`
- Fusion model: modality-specific projection layers, multi-head cross-attention, gated fusion, Cox-style risk head with zero-mask handling for missing modalities
- Demo enhancement: per-modality risk predictions are now extracted from the trained verifier projection layers for clinical explanation in Streamlit

## Known Limitations

- Full TCGA slide extraction is still in progress; extraction coverage is not yet 100%
- The current refreshed aligned cohort has no populated event labels, so the verifier cannot yet produce trustworthy held-out outcome metrics on the latest crosswalk
- The current full-model Slurm path now fails fast when CUDA is unavailable, which prevents bad training but also means a clean GPU-enabled rerun is still required
- C-index remains skipped because `event_observed` is `0` for all evaluated cases in the current artifact
- Literature evidence is deployment-ready conceptually but not trained per patient, by design
- The demo currently relies on exported artifacts rather than live inference

## Phase 2 Targets

- Complete TCGA UNI2 extraction to full 1,058-slide coverage
- Repair aligned endpoint coverage so the refreshed cohort contains both event and censored cases
- Re-run the full Cox verifier on a GPU-enabled Isambard environment
- Add SurvPath pathway tokenization for genomics
- Extend external validation to CPTAC-BRCA
- Profile inference on Lenovo and Intel-aligned deployment hardware
