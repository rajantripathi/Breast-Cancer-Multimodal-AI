# TCGA Ablation Results

## Final Frozen Ablation Table

- Model family: `Simple late fusion`
- Endpoint: `PFI`
- Evaluation: `5-fold stratified cross-validation`
- Cohort: `788` aligned `TCGA-BRCA` patients
- Genomics: `50 Hallmark pathways`

| Modalities | C-index (mean +/- std) |
| --- | --- |
| `V` | `0.534 +/- 0.072` |
| `V+C` | `0.526 +/- 0.063` |
| `V+G` | `0.601 +/- 0.046` |
| `V+C+G` | `0.589 +/- 0.060` |

## Clean Scientific Readout

- `V+G` is the best-performing configuration.
- Adding clinical features does not improve the best pathology-plus-genomics baseline in the current BRCA/PFI setup.
- Full three-modality fusion is competitive but slightly weaker than `V+G`.
- This indicates that pathway-level genomics is the strongest complementary signal to pathology in the current pipeline.

## Interpretation

- The simplest scientifically defensible conclusion is:
  - `Vision + Genomics` is the best current multimodal configuration
  - `Clinical` is not yet contributing stable marginal value
- This is a much cleaner and more credible result than the earlier unstable cross-attention fusion experiments.

## Why This Table Matters

- It provides the standard multimodal ablation expected in published work.
- It directly compares all modality combinations under the same:
  - endpoint
  - cohort
  - CV protocol
  - genomics representation
- It also shows that added architectural complexity is not automatically better.

## Final Proposal Position

- Use this table in the technical appendix.
- Lead with:
  - `PFI`
  - `5-fold CV`
  - `V+G = 0.601 +/- 0.046`
- Present `V+C+G` as a near-best multimodal configuration, but not the headline result.
