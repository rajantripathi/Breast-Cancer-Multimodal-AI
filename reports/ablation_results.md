# TCGA Ablation Results

## Cohort

- Aligned patients: `689`
- Vision patients with embeddings: `693`
- Genomics patients with tensors: `1094`
- Clinical patients with rows: `1097`
- Current aligned clinical endpoint coverage:
  - `vital_status_blank`: `689`
  - `days_to_death_present`: `0`
  - `days_to_last_followup_present`: `614`

## Modality Comparison

| Run | Modalities | Validation Accuracy | Split Status |
| --- | --- | --- | --- |
| Full model | Vision + Clinical + Genomics | unavailable | failed before training |
| V only | Vision | `0.3875` | single-class cohort, no held-out split |
| V + C | Vision + Clinical | `0.2380` | single-class cohort, no held-out split |
| V + G | Vision + Genomics | `0.8578` | single-class cohort, no held-out split |

## Current Readout

- The corrected full-model verifier job `3247553` failed immediately with `AssertionError: CUDA not available`, which is the intended fail-fast behavior after adding the GPU guard.
- The three ablation jobs completed, but each reported `num_train=689`, `num_val=0`, and `num_test=0`.
- This is not a split bug anymore. The aligned cohort currently has no populated event labels, so `_split_frame()` correctly falls back to a no-split path.
- Because the aligned set is single-class, these ablation accuracies are not proposal-grade comparative metrics and should not be presented as final evidence of multimodal benefit.

## Blocking Issue

- The current aligned TCGA subset lacks usable endpoint supervision for deceased cases.
- Without non-empty `vital_status` or `days_to_death` values in the aligned cohort, neither binary outcome evaluation nor Cox-style survival evaluation can produce scientifically valid held-out metrics.

## Follow-Up

- Repair the aligned clinical endpoint mapping before any further verifier retraining.
- Rebuild the TCGA crosswalk only after confirming the clinical merge preserves outcome labels for the aligned patient subset.
- Re-run the full model and ablations once the aligned cohort contains both censored and event cases.
