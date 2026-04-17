# Mammography Results Summary

## Canonical Benchmark

The current mammography benchmark remains the recovered VinDr-only
legacy recipe.

- Run: `outputs/mammography/legacy_recovery/summary.json`
- Dataset: `VinDr-Mammo` test set only
- `best_epoch`: `32`
- `best_val_auroc`: `0.7560115949667303`
- `test_auroc`: `0.7407175925925926`

This is the reference result for the screening layer on
`feature/mammography-screening`.

## CMMD Auxiliary Ablation

CMMD was added as a train-time auxiliary dataset and evaluated against
the same VinDr test benchmark.

- Run: `outputs/mammography/vindr_cmmd_legacy_seed42/summary.json`
- Data sources in training:
  - `vindr`: `4996`
  - `cmmd`: `826`
- `best_epoch`: `32`
- `best_val_auroc`: `0.7201704026035763`
- `test_auroc`: `0.7091564427808968`

## Decision

`VinDr + CMMD` underperformed the recovered VinDr-only legacy baseline:

- VinDr-only legacy: `0.7407`
- VinDr+CMMD: `0.7092`

Because the auxiliary-data run did not improve the canonical VinDr test
metric, CMMD is retained as an experimental ablation path only. The
final default training path remains the VinDr-only legacy recipe.
