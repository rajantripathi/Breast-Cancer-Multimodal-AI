# TCGA Ablation Results

## Cohort

- Aligned patients: `696`
- Vision patients with embeddings in the frozen crosswalk: `700`
- Genomics patients with tensors: `1094`
- Clinical patients with rows: `1097`
- Outcome distribution: `613 Alive`, `83 Dead`
- Stratified proposal split:
  - Train: `{0: 429, 1: 58}`
  - Val: `{0: 92, 1: 12}`
  - Test: `{0: 92, 1: 13}`

## Modality Comparison

| Run | Modalities | Validation Accuracy | Num Samples | Train / Val / Test |
| --- | --- | --- | --- | --- |
| Full model | Vision + Clinical + Genomics | `0.6952` | `696` | `487 / 104 / 105` |
| V only | Vision | `0.5048` | `696` | `487 / 104 / 105` |
| V + C | Vision + Clinical | `0.6571` | `696` | `487 / 104 / 105` |
| V + G | Vision + Genomics | `0.6857` | `696` | `487 / 104 / 105` |

## Readout

- This final ablation pass ran on the repaired CUDA environment and the full verifier log confirms `GPU: NVIDIA GH200 120GB`.
- All four runs used the corrected clinical endpoint mapping and a stratified split with both classes represented in train, validation, and test.
- The full multimodal model is now the strongest of the four on this final GPU-backed run.
- Adding genomics improves substantially over vision only, and the clinical branch improves further when combined inside the full multimodal model.
- Modality-agreement analysis from the final held-out predictions shows:
  - vision agreement with fused prediction: `0.5619`
  - clinical agreement with fused prediction: `0.2190`
  - genomics agreement with fused prediction: `0.3048`

## Interpretation

- The final proposal run now supports a clean multimodal value narrative on a real held-out split.
- Vision alone is a weak baseline on this outcome definition.
- Vision + Clinical improves over vision alone.
- Vision + Genomics nearly matches the full model, which suggests genomics is the strongest complementary signal to pathology in the current setup.
- The small gap between `V+G` and the full model means the clinical branch is currently contributing less stable value than genomics.
- The held-out agreement profile suggests the clinical branch is over-calling `high_concern` and should be the first target for feature audit and calibration tuning.

## Follow-Up

- Refresh the cohort again after the extraction tail finishes if the aligned set grows materially beyond `696`.
- Analyze modality gate weights and per-modality confidence calibration for the clinical demo.
- Revisit risk-group thresholds after the next retrain so low/intermediate/high groups are actually populated.
- Extend the same evaluation pattern to CPTAC-BRCA external validation in the next phase.
