# Mammography Screening Agent

Screening layer for breast cancer detection from full-field digital
mammograms. Designed as the front-end of the two-stage breast cancer
AI platform (screening + multimodal diagnosis/prognosis).

## Dataset
- VinDr-Mammo (PhysioNet): 5,000 exams, 20,000 images, multi-reader BI-RADS
- Access: https://physionet.org/content/vindr-mammo/1.0.0/
- Optional auxiliary source: CBIS-DDSM, integrated as train-only metadata-compatible input
- Preferred next external cohort: EMBED Open Data, staged through a tables-first
  download path and processed into the same image-level metadata contract

## Architecture
- Input: 4-view mammogram (L-CC, R-CC, L-MLO, R-MLO)
- Encoder: ConvNeXt-Base pretrained on ImageNet (fine-tuned)
- Aggregation: View-level encoding -> attention-weighted exam prediction
- Output: Exam-level suspicion score (0.0 to 1.0)

## Metrics
- AUROC (primary)
- Sensitivity at 90% specificity
- Specificity at 90% sensitivity
- External metadata-compatible cohorts can be evaluated with `agents.mammography.evaluation.evaluate_screener --model-type standard`

## Current Result
- Canonical benchmark: VinDr-only legacy recovery
- `test_auroc`: `0.7407175925925926`
- CMMD auxiliary ablation was completed and did not improve VinDr test
  performance (`0.7091564427808968`), so it is not the default path

## Auxiliary Data
- The non-legacy screener can ingest additional image-level metadata CSVs via `--aux-metadata-csv`
- Auxiliary datasets must use the same metadata contract as VinDr and include `dataset_source`
- Validation and test remain VinDr-only so benchmark reporting stays comparable
- Source-aware train-fit intensity harmonization is available via `--harmonization-method source_percentile`

## EMBED External Validation
- `data.preprocess.download_embed` downloads the EMBED table bundle first and
  then syncs only the selected DICOM objects referenced by a manifest.
- `agents.mammography.preprocessing.prepare_embed` converts EMBED clinical and
  metadata tables into `metadata.csv` plus `download_manifest.txt`.
- Recommended first pass:
  - `exam_type=screening`
  - `label_mode=recall_or_pathology`
  - `preferred_image_type=2D`
  - `allow_cview_fallback=true`
- Slurm launchers:
  - `slurm/mammography/50_download_embed_tables.sh`
  - `slurm/mammography/51_prepare_embed_external.sh`
  - `slurm/mammography/52_download_embed_images.sh`
  - `slurm/mammography/53_evaluate_screener_embed_external.sh`
