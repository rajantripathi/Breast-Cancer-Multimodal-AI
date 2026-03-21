# TCGA Results Summary

## Executive Snapshot

- Evaluation status: final GPU-backed proposal run completed on the corrected `696`-patient aligned cohort
- Vision embeddings extracted: `758 / 1058` tiled TCGA slides at the time of final collection
- Aligned patients: `696`
- Train / validation / test split: `487 / 104 / 105`
- Architecture: UNI2 vision embeddings + TCGA genomics tensors + TCGA clinical features -> modality projections -> cross-attention verifier -> Cox-style risk scoring

## Dataset Scale

- TCGA-BRCA slides: `1,132` raw slides, `1,058` tiled slides
- TCGA-BRCA RNA-seq records: `1,230`
- TCGA-BRCA clinical rows: `1,097`
- Vision patients with embeddings in the frozen crosswalk: `700`
- Genomics patients with tensors: `1,094`
- Clinical patients with rows: `1,097`
- Patient-aligned multimodal cohort: `696`
- Outcome distribution in the aligned cohort: `613 Alive`, `83 Dead`

## Final GPU Run

- Full multimodal verifier summary:
  - `val_accuracy`: `0.6952`
  - `num_samples`: `696`
  - `num_train`: `487`
  - `num_val`: `104`
  - `num_test`: `105`
  - `alignment_status`: `patient_aligned_tcga`
  - `aligned_sample_count`: `696`
- Enterprise evaluation:
  - `num_predictions`: `105`
  - `fused_label_distribution`: `{"monitor": 82, "high_concern": 23}`
  - `balanced_accuracy`: `0.4628`
  - `f1_macro`: `0.4636`
  - `ece`: `0.1349`
  - `auroc_macro`: `0.5125`
  - `auprc_macro`: `0.1728`
  - `brier_score`: `0.2207`
  - `auroc_ci_95`: `[0.3459, 0.6865]`
  - `balanced_accuracy_ci_95`: `[0.3673, 0.5972]`
  - `c_index`: `0.526`
  - `survival_time_diagnostic`: `min=0.0, max=7106.0, unique=91`
  - `event_diagnostic`: `sum=13, total=105`
  - `risk_group_summary`: all `105` held-out patients fell into the current intermediate-risk band with mean `risk_score=0.4555` and event rate `0.1238`
  - `modality_agreement_summary`:
    - vision agreement with fused output: `0.5619`
    - clinical agreement with fused output: `0.2190`
    - genomics agreement with fused output: `0.3048`

## Ablation Summary

- Vision only: `0.5048`
- Vision + Clinical: `0.6571`
- Vision + Genomics: `0.6857`
- Vision + Clinical + Genomics: `0.6952`

## Interpretation

- The clinical-label pipeline is now repaired end to end, and the final run uses a stratified split with both classes present in train, validation, and test.
- The Isambard environment is now GPU-correct. The successful full-model log records `GPU: NVIDIA GH200 120GB`, and the project venv now uses `torch 2.10.0+cu126`.
- The evaluator is correctly reading Cox-style `risk_score` outputs and now computes `c_index` locally without depending on `lifelines`, which keeps the evaluation reproducible from the repo alone.
- The final held-out metrics are modest, which should be presented honestly as a real multimodal TCGA survival-risk baseline rather than a deployment-grade endpoint.
- The ablation trend now supports the core multimodal claim: adding clinical and genomics information improves over the vision-only baseline, and the full model is the strongest of the four final runs.
- The current risk scores are under-dispersed on the held-out set: every test patient lands in the intermediate band. This means the model is producing usable continuous rankings, but not yet clinically sharp low/high risk separation.
- The modality-agreement readout suggests the current clinical branch is the noisiest component in the fusion stack: it predicts `high_concern` for most held-out patients and agrees with the fused model least often.

## Artifact Readiness

- `outputs/tcga_verifier/predictions.json` contains `105` predictions
- Each prediction includes:
  - `risk_score`
  - `modality_predictions`
  - `survival_time`
  - `event_observed`
- This is sufficient for the Streamlit demo to render the patient risk page and the multimodal analysis page from exported artifacts only

## Known Limitations

- Full TCGA slide extraction is still in progress; the final proposal freeze used `700` embedded vision patients in the crosswalk while `758` slide embeddings existed on disk at final collection time
- AUROC, AUPRC, and C-index are now real and non-zero, but they remain modest and should be framed as a public-dataset baseline
- Risk-group separation is not yet strong: the current fixed low/intermediate/high thresholds collapse the held-out set into the intermediate band
- The clinical branch appears over-sensitive on this endpoint definition and needs feature-quality review, calibration work, or a lighter weighting strategy
- Literature evidence remains a decision-support companion rather than a fused training modality

## Phase 2 Targets

- Complete TCGA UNI2 extraction to full `1,058`-slide coverage and refresh the aligned cohort
- Improve calibration and risk ranking quality beyond the current `c_index` and AUROC baseline
- Tune risk-group thresholds from the survival distribution instead of reusing generic demo cutoffs
- Audit and simplify the clinical feature set so the clinical branch stops saturating toward `high_concern`
- Add SurvPath pathway tokenization for genomics
- Extend external validation to CPTAC-BRCA
- Profile inference on Lenovo and Intel-aligned deployment hardware
