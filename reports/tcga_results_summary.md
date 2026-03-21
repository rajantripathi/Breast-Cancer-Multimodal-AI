# TCGA Results Summary

## Executive Snapshot

- Current science focus: pathway-based genomics plus fixed-window survival prediction on TCGA-BRCA
- Current pathway-aligned cohort: `789` patients before endpoint filtering
- Current 3-year survival cohort after censoring rules: `353`
- Current model family: UNI2 vision embeddings + Hallmark pathway genomics + TCGA clinical features -> modality projections -> cross-attention verifier -> Cox-style risk scoring
- Current conclusion: the survival pipeline is scientifically cleaner than before, but multimodal fusion is not yet stable enough to lead the proposal as a robust result

## Dataset Scale

- TCGA-BRCA slides with current UNI2 embeddings on disk: `758`
- Pathway-genomics patients: `1094`
- Clinical rows: `1097`
- Pathway-aligned patients before endpoint filtering: `789`
- Usable patients for the `1095`-day endpoint: `353`
- Outcome filtering effect:
  - most excluded patients are alive but censored before `3` full years of follow-up

## Current Best-Calibrated 3-Year Pathway Run

- Full multimodal summary:
  - `val_accuracy`: `0.3396`
  - `num_samples`: `353`
  - `num_train`: `247`
  - `num_val`: `53`
  - `num_test`: `53`
- Classification threshold selected on validation:
  - `0.536387`
- Primary survival metrics:
  - `C-index`: `0.4372`
  - `5yr AUROC`: `0.6011`
  - `3yr AUROC`: `0.6167`
  - `Risk group separation`: `p=0.0054`
- Secondary binary metrics:
  - `balanced_accuracy`: `0.5597`
  - `f1_macro`: `0.3358`
  - `auroc_macro`: `0.6167`
  - `auprc_macro`: `0.2296`
  - `brier_score`: `0.2916`
  - `ece`: `0.2207`

## Risk-Group Readout

- Tertile survival groups:
  - low risk: `17`, events `4`, event rate `0.2353`, median survival `2311.0`
  - mid risk: `18`, events `10`, event rate `0.5556`, median survival `1478.5`
  - high risk: `18`, events `3`, event rate `0.1667`, median survival `2234.0`
- This produces a statistically significant log-rank result, but the tertile ordering is not biologically clean.
- Interpretation:
  - the model is separating groups in some way
  - but the current fused ranking is not monotonic enough to claim clinically intuitive risk stratification

## True Ablation Status

- Corrected single-run ablation:
  - `V`: `0.4528`
  - `V + C`: `0.6981`
  - `V + G`: `0.6038`
  - `V + C + G`: `0.4151`
- Calibrated rerun:
  - `V`: `0.6226`
  - `V + C`: `0.2264`
  - `V + G`: `0.3208`
  - `V + C + G`: `0.3396`

## Seed Stability Audit

- Seed sweep on the same `353`-patient 3-year setup:

| Seed | Full | V | V + C | V + G |
| --- | --- | --- | --- | --- |
| `7` | `0.6038` | `0.7170` | `0.3962` | `0.6981` |
| `13` | `0.4340` | `0.4151` | `0.3962` | `0.6415` |
| `23` | `0.4906` | `0.4151` | `0.7170` | `0.4151` |

- Scientific interpretation:
  - rankings flip across seeds
  - the full fused model is not consistently best
  - the current fusion objective is unstable under random initialization

## What Has Improved Scientifically

- Endpoint definition is now methodologically correct for fixed-window survival labeling.
- Cox loss and survival-focused metrics are now primary.
- Pathway-based genomics is implemented and reproducible.
- Evaluation can now report:
  - `C-index`
  - time-dependent AUROC
  - risk-group survival summaries
  - log-rank p-values

## What Remains Weak

- The fused multimodal model is unstable across seeds.
- Binary accuracy-based comparisons are unreliable on their own.
- Current risk scores remain compressed and poorly calibrated in several runs.
- The present fusion stack is not yet robust enough to support a strong “more modalities is better” claim.

## Most Defensible Proposal Position

- Present the project as:
  - a real multimodal TCGA survival-risk platform
  - with corrected methodology
  - and a working pathway-genomics extension
- State clearly that:
  - pathway survival modeling is promising
  - but the present multimodal fusion remains sensitive to initialization
  - and therefore repeated-run robustness work is the next scientific priority

## Recommended Next Step

- Do not keep tuning the current full fusion result with minor changes.
- Freeze the honest science finding:
  - the current fusion configuration is unstable
- Next rigorous experiment:
  - repeated-run reporting with mean/std for `V`, `V + C`, `V + G`, and full fusion
  - then decide which configuration is truly robust enough to carry forward
