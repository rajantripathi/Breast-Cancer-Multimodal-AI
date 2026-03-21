# TCGA Ablation Results

## Current Scientific Readout

- Current methodology under test:
  - Hallmark pathway genomics (`50` features)
  - Fixed-window survival endpoint at `1095` days (`3` years)
  - Cox loss for training
  - Survival-first evaluation
- Current pathway-aligned cohort: `353`
- Split: `247 / 53 / 53`
- Class balance:
  - Train: `{0: 212, 1: 35}`
  - Val: `{0: 46, 1: 7}`
  - Test: `{0: 45, 1: 8}`

## Single-Run Ablation

| Run | Modalities | Validation Accuracy |
| --- | --- | --- |
| Full model | Vision + Clinical + Genomics | `0.4151` |
| V only | Vision | `0.4528` |
| V + C | Vision + Clinical | `0.6981` |
| V + G | Vision + Genomics | `0.6038` |

## Seed Stability Sweep

| Seed | Full | V | V + C | V + G |
| --- | --- | --- | --- | --- |
| `7` | `0.6038` | `0.7170` | `0.3962` | `0.6981` |
| `13` | `0.4340` | `0.4151` | `0.3962` | `0.6415` |
| `23` | `0.4906` | `0.4151` | `0.7170` | `0.4151` |

## Stability Interpretation

- The current 3-year pathway setup is not stable across random seeds.
- Modality rankings flip across seeds:
  - `V` is strongest for seed `7`
  - `V + G` is strongest for seed `13`
  - `V + C` is strongest for seed `23`
- The full fused model is never the strongest run in the current seed sweep.
- This means the main issue is not only thresholding. Training variance is high enough that the fusion claim is not yet robust.

## What The Results Still Support

- The pathway-based genomics pipeline is operational and scientifically usable.
- The fixed-window survival setup is more methodologically defensible than the older overall-survival label shortcut.
- Survival-oriented metrics can now be computed consistently from the exported artifacts.

## What The Results Do Not Yet Support

- A strong claim that `Vision + Clinical + Genomics` is reliably better than simpler baselines.
- A claim that genomics is always the strongest complement to pathology.
- A claim that current multimodal fusion is stable enough for publication-quality comparative conclusions.

## Most Defensible Proposal Framing

- Present the TCGA pathway-survival work as a promising but unstable multimodal research track.
- Do not use the current full fused model as the headline scientific result.
- Treat the seed sweep as an honest robustness audit showing that the present fusion objective is sensitive to initialization.

## Next Scientific Step

- Freeze the current finding:
  - pathway genomics and fixed-window survival are viable
  - fusion is unstable
- Move the next experiment to repeated-run reporting for the simpler baselines:
  - `V`
  - `V + G`
  - `V + C`
- Report mean and standard deviation across seeds before making any comparative multimodal claim.
