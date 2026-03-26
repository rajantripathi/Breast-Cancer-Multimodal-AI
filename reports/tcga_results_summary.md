# TCGA Results Summary

## Final Paper Benchmark

- Dataset: `TCGA-BRCA`
- Endpoint: `Progression-Free Interval (PFI)`
- Endpoint source: `TCGA-CDR (Liu et al., Cell 2018)`
- Evaluation: `5-fold stratified cross-validation`
- Slide coverage:
  - `UNI2`: `1054 / 1054`
  - `CTransPath`: `1049 / 1054`
  - `CONCH`: `1049 / 1054`
- Aligned cohorts:
  - `UNI2`: `1049`
  - `CTransPath`: `1044`
  - `CONCH`: `1044`

## Encoder Comparison

| Encoder | Cross-attention V+G | Simple concat V+G |
| --- | --- | --- |
| `UNI2` | `0.5648 +/- 0.0295` | `0.5369 +/- 0.0593` |
| `CTransPath` | `0.5787 +/- 0.0387` | `0.4846 +/- 0.0303` |
| `CONCH` | `0.5846 +/- 0.0587` | `0.4798 +/- 0.0350` |

## Best Model

- Best encoder family under the refreshed benchmark: `CONCH`
- Best full configuration: `CONCH + Cross-Attention + V+C+G`
- Headline result: `C-index = 0.6093 +/- 0.0441`

## Risk Stratification

- Kaplan-Meier source: pooled `5-fold CV` predictions from `conch_ca_vcg`
- Tertile log-rank p-value: `0.0412`
- Group event rates:
  - `low_risk`: `40 / 347` (`0.1153`)
  - `mid_risk`: `43 / 348` (`0.1236`)
  - `high_risk`: `54 / 348` (`0.1552`)

## Interpretation

- Full slide coverage changed the benchmark outcome materially from earlier frozen proposal numbers.
- In the final paper benchmark, pathology-specialized encoders outperform `UNI2` under cross-attention.
- `CONCH` is the strongest encoder and benefits from adding both clinical and genomics signals in the final ablation.
- Pooled out-of-fold Kaplan-Meier analysis now shows statistically significant but still modest tertile separation, so the paper can claim real risk stratification while keeping the emphasis on benchmark rigor rather than strong clinical utility.
