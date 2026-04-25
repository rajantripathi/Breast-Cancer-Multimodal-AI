# Paper 2 Statistical Depth Status

## Current Status

### Stage 2: Ready from tracked artifacts

The following analyses can be computed directly from the tracked
`outputs/paper/*/artifact.json` files, which contain the full
cross-validation prediction sets:

- B1. Pairwise encoder significance
- B2. Ablation significance tests
- B3. Time-dependent AUC at fixed horizons
- B4. Survival calibration bins
- B5. Alternative risk stratification (median / quartiles)

These analyses do not require Isambard, GPU access, or model retraining.

### Stage 1: Requires one inference-only Isambard run

The tracked repo contains only the final mammography aggregate summary:

- `outputs/mammography/summary.json`

It does not contain:

- a mammography checkpoint
- per-exam test predictions

So A1, A2, and A4 require one inference-only Isambard evaluation against the
final legacy checkpoint to regenerate:

- `reports/paper2/stage1_statistics/predictions.json`
- `reports/paper2/stage1_statistics/stage1_statistical_depth.json`

## Completed Follow-up

### A3. Breast-density subgroup

Density retention support has now been added to
`agents/mammography/preprocessing/prepare_vindr.py`.

The updated preprocessing path writes:

- `breast_density`
- `exam_density`

alongside the existing metadata fields.

VinDr preprocessing was re-run on Isambard, refreshed `metadata.csv` was
generated with `exam_density`, and the saved Stage 1 predictions were
stratified by density.

Current subgroup summary:

- Density B: `n=73`, positives `5`, AUROC `0.8265`
- Density C: `n=564`, positives `24`, AUROC `0.7257`
- Density D: `n=113`, positives `1`, AUROC `0.7054`

The density-D subgroup contains only one positive exam, so that estimate is
descriptive rather than stable.

## Expected Reporting Caveat

For B1 and B2, inferential tests operate on 5 paired CV folds. This gives very
limited statistical power. Exact sign-flip / permutation p-values are preferred
to over-interpreting Wilcoxon alone, and non-significant p-values should be
presented as a power limitation rather than evidence of no effect.
