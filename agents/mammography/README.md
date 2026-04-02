# Mammography Screening Agent

Screening layer for breast cancer detection from full-field digital
mammograms. Designed as the front-end of the two-stage breast cancer
AI platform (screening + multimodal diagnosis/prognosis).

## Dataset
- VinDr-Mammo (PhysioNet): 5,000 exams, 20,000 images, multi-reader BI-RADS
- Access: https://physionet.org/content/vindr-mammo/1.0.0/

## Architecture
- Input: 4-view mammogram (L-CC, R-CC, L-MLO, R-MLO)
- Encoder: ConvNeXt-Base pretrained on ImageNet (fine-tuned)
- Aggregation: View-level encoding -> attention-weighted exam prediction
- Output: Exam-level suspicion score (0.0 to 1.0)

## Metrics
- AUROC (primary)
- Sensitivity at 90% specificity
- Specificity at 90% sensitivity

