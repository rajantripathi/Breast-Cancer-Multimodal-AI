# TCGA Ablation Results

## Cohort

- Aligned patients: `681`
- Train / validation / test: `476 / 102 / 103`
- Vision patients with embeddings: `684`
- Genomics patients with tensors: `1094`
- Clinical patients with rows: `1097`

## Modality Comparison

| Run | Modalities | Validation Accuracy |
| --- | --- | --- |
| Full model | Vision + Clinical + Genomics | `0.5631` |
| V only | Vision | `0.0485` |
| V + C | Vision + Clinical | `0.6408` |
| V + G | Vision + Genomics | `0.8932` |

## Current Readout

- The strongest current ablation is `Vision + Genomics`, which substantially outperforms the full fusion run.
- The current full-model training log shows `CUDA requested but unavailable; falling back to CPU`, followed by flat `0.0000` train and validation losses across epochs.
- This means the current full-model underperformance is not yet evidence that clinical features are intrinsically harmful. The run quality is confounded by the fallback execution path.
- The evaluation pipeline also needed to be updated for Cox-style `risk_score` predictions so AUROC, fused-label distribution, and survival diagnostics reflect the new output format.

## Follow-Up

- Re-run enterprise evaluation after the Cox-aware evaluator patch.
- Inspect why the Slurm training environment fell back to CPU despite requesting `--device cuda`.
- If a clean GPU run still shows degradation from clinical features, reduce clinical influence during fusion or regularize the clinical projection more aggressively.
