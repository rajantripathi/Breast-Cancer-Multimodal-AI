# Experiment Log

## 2026-03-10

- Implemented Phase 1 vision registry and feature extraction scaffold.
- Kept local validation lightweight; no heavy downloads or feature extraction were run locally.
- Target deployment path remains Isambard, with TCGA-BRCA as the primary extraction dataset and BreaKHis as the benchmark fallback.
- Expanded the verifier dataset builder to encode modality confidence, source provenance, missing-modality cases, and mixed-consensus bundles.
- Local verifier regeneration produced 2,000 bundled rows, balanced between `monitor` and `high_concern`.
- Local verifier retraining wrote an artifact with `num_samples=1400` and `val_accuracy=0.9579` on the train split, which is useful as a pipeline check but not a clinical or held-out metric.
