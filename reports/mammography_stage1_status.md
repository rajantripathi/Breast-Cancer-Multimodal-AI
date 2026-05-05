# Mammography Stage 1 Status

## Current Benchmark

- Reference benchmark: `outputs/mammography/summary.json`
- Reference provenance match: `outputs/mammography/repro_legacy224_seed42/summary.json`
- Headline result: `test_auroc = 0.7407`

## Repro and Recovery Runs

| Run | Training path | Test AUROC | Outcome |
| --- | --- | --- | --- |
| `repro_baseline_224_seed42` | non-legacy screener, 224px repro | `0.5421` | weak control |
| `recovery_seed42` | non-legacy screener, weighted BCE + balanced sampling | `0.6576` | improved over weak control, still below reference |
| `vindr_cbis_aux_seed42_rerun` | non-legacy screener + CBIS auxiliary data | `0.5453` | negative result; no benchmark gain |
| `repro_legacy224_seed42` | legacy screener path | `0.7407` | current reference |

## Interpretation

- The retained screening benchmark is still the legacy 224px path.
- The non-legacy recovery configuration did complete cleanly after the PNG cache fix, but it did not recover the historical result.
- The CBIS auxiliary-data run did not help: only `105` CBIS exams survived into the 4-view train split, and the original mixed-source path overweighted that small auxiliary set.
- The gap is large enough that the next compute cycle should stay VinDr-first and isolate optimization behavior before revisiting auxiliary data.

## Next Recovery Step

- Keep `VinDr-Mammo` as the default benchmark and training source.
- Use one-variable VinDr-only ablations around the completed `recovery_seed42` run:
  - weighted BCE without balanced sampling: `slurm/mammography/44_train_screener_vindr_weighted_only.sh`
  - balanced sampling without weighted BCE: `slurm/mammography/45_train_screener_vindr_sampler_only.sh`
- Keep CBIS as a diagnostic ablation only.
- Launch path:
  - train equal-weight mixed-source diagnostic: `slurm/mammography/42_train_screener_vindr_cbis_aux.sh`

## Important Assumption

- `CBIS-DDSM` is treated as lesion-enriched auxiliary training data, not as a benchmark dataset.
- Any run only replaces the current benchmark if it improves `VinDr` test performance over `0.7407`.
