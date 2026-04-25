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

## Confirmed Limitation

### A3. Breast-density subgroup

This is out of scope for the current paper-ready pass.

`agents/mammography/preprocessing/prepare_vindr.py` currently saves:

- `study_id`
- `image_id`
- `laterality`
- `view`
- `label`
- `split`
- `png_path`

It does not preserve `breast_density`, so subgrouping by density would require:

1. modifying preprocessing,
2. re-running VinDr preprocessing, and
3. re-running Stage 1 evaluation.

Recommendation: treat A3 as supplementary or future work unless specifically
requested by reviewers.

## Expected Reporting Caveat

For B1 and B2, inferential tests operate on 5 paired CV folds. This gives very
limited statistical power. Exact sign-flip / permutation p-values are preferred
to over-interpreting Wilcoxon alone, and non-significant p-values should be
presented as a power limitation rather than evidence of no effect.
