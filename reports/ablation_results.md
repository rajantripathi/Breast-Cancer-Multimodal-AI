# TCGA Ablation Results

## Final Best-Encoder Ablation

- Encoder: `CONCH`
- Fusion: `Cross-attention`
- Endpoint: `PFI`
- Evaluation: `5-fold stratified cross-validation`
- Cohort: `1043` aligned `TCGA-BRCA` patients
- Genomics: `50 Hallmark pathways`

| Modalities | C-index (mean +/- std) |
| --- | --- |
| `V` | `0.5675 +/- 0.0461` |
| `V+C` | `0.5615 +/- 0.0529` |
| `V+G` | `0.5846 +/- 0.0587` |
| `V+C+G` | `0.6093 +/- 0.0441` |

## Readout

- The best encoder in the refreshed benchmark is `CONCH`.
- Unlike the earlier frozen proposal phase, the final rebuilt benchmark shows that full three-modality fusion is strongest for the best encoder.
- `V+C+G` is the headline configuration for the paper benchmark.
- `V+G` remains competitive, but it is not the final best model after 100% UNI2 coverage and the encoder refresh.

## Encoder Context

| Encoder | Cross-attention V+G | Simple concat V+G |
| --- | --- | --- |
| `UNI2` | `0.5648 +/- 0.0295` | `0.5369 +/- 0.0593` |
| `CTransPath` | `0.5787 +/- 0.0387` | `0.4846 +/- 0.0303` |
| `CONCH` | `0.5846 +/- 0.0587` | `0.4798 +/- 0.0350` |

## Interpretation

- The encoder study and the ablation together indicate that encoder choice matters more than previously assumed.
- In the rebuilt paper benchmark, pathology-specialized vision features benefit the cross-attention model more than the simple concat baseline.
- The strongest result in the completed study is:
  - `CONCH + Cross-Attention + Vision + Clinical + Genomics`
  - `C-index 0.6093 +/- 0.0441`
