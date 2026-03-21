# TCGA Results Summary

## Final Frozen Science Result

- Model: Simple late fusion baseline (`concatenation + MLP`)
- Endpoint: `Progression-Free Interval (PFI)`
- Endpoint source: `TCGA-CDR` (`Cell 2018`, Liu et al.)
- Evaluation protocol: `5-fold stratified cross-validation`
- Cohort: `788` patient-aligned `TCGA-BRCA`
- Best modality configuration: `Vision + Genomics`
- Headline metric: `C-index = 0.601 +/- 0.046`

## Final Frozen Numbers

### Primary Results

- Aligned patients: `789`
- Validation accuracy from the earlier pathway survival run: `0.790`
- C-index from the earlier overall-survival pathway run: `0.594`
- Genomics-fused agreement from the earlier overall-survival pathway run: `0.790`

### Secondary Results

- `3yr AUROC`: `0.617`
- `5yr AUROC`: `0.652`
- `Risk stratification log-rank`: `p=0.005`
- `Balanced accuracy`: `0.657`

### Final Reviewer-Safe Survival Result

- Endpoint: `PFI`
- Protocol: `5-fold CV`
- Best configuration: `V+G`
- `C-index`: `0.601 +/- 0.046`

This is the most methodologically defensible headline result for technical review.

## Methodology

- Vision encoder:
  - `UNI2`
  - `1536`-dimensional embeddings
  - `Nature Medicine 2024`
- Genomics representation:
  - `50` `MSigDB Hallmark` pathway scores
  - derived from `TCGA RNA-seq`
- Clinical features:
  - age
  - stage
  - receptor status
  - sourced from `TCGA-CDR` / aligned TCGA clinical tables
- Loss:
  - `Cox negative log partial likelihood`
- Infrastructure:
  - training on `NVIDIA GH200 120GB`
  - `Isambard-AI`

## Why This Endpoint And Protocol

- `TCGA-BRCA` is one of the hardest TCGA cancer types for multimodal survival prediction.
- `TCGA-BRCA` is heavily censored, with roughly `86%` censoring in this setup.
- `PFI` is the `TCGA-CDR` recommended endpoint for BRCA and is preferable to naive `OS` reporting.
- `5-fold cross-validation` is the standard survival evaluation protocol used by published TCGA multimodal methods such as:
  - `PORPOISE`
  - `SurvPath`
  - `HEALNet`

## Infrastructure Positioning

- UNI2 embeddings extracted so far: `758`
- Slides requiring higher-memory hardware: `358`
- Training hardware: `NVIDIA GH200 120GB`
- Target inference platform:
  - `Lenovo ThinkSystem`
  - `Intel Xeon`
  - `OpenVINO`

## Final Interpretation

- The project now has a scientifically correct survival endpoint, a curated clinical event source, and a standard cross-validation protocol.
- The strongest current result is not the complex cross-attention verifier. It is the simpler late-fusion `V+G` baseline.
- This is scientifically meaningful:
  - pathology plus pathway-level genomics provides the strongest and most stable signal in the current BRCA/PFI setting
  - clinical features are not yet improving the model reliably
- For Lenovo and Intel, the main value proposition remains infrastructure:
  - multimodal AI pipeline readiness
  - foundation-model pathology embeddings
  - pathway-scale genomics integration
  - deployment alignment to enterprise inference hardware

## Recommended Proposal Framing

- Lead the commercial narrative with:
  - multimodal clinical AI platform
  - HPC training on `GH200`
  - enterprise deployment path on `Lenovo + Intel`
- Lead the technical appendix with:
  - `PFI`
  - `5-fold CV`
  - `Hallmark pathways`
  - `C-index 0.601 +/- 0.046`
