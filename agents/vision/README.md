## Vision Aggregation

The pathology vision stack supports three slide-level aggregation modes:

- `mean`: default pooled baseline for paper-facing runs
- `abmil`: attention-based multiple instance learning, opt-in
- `transmil`: transformer-style multiple instance learning, opt-in

The Stage 2 aggregation sweep on `CONCH + V+C+G + flat genomics + flat clinical`
keeps `mean` as the default. The decision is based on the Phase 1 seed sweep:

- `verifier_mean`: `0.6052`
- `verifier_abmil`: `0.6053`
- `verifier_transmil`: `0.6307`
- `simple_fusion_mean`: `0.6125`
- `simple_fusion_abmil`: `0.5735`
- `simple_fusion_transmil`: `0.5710`

Interpretation:

- `ABMIL` did not improve on the verifier mean-pooling baseline.
- `TransMIL` was promising in the verifier path, but the gain did not clear the
  variance threshold strongly enough to adopt it as the new default.
- Both MIL variants regressed relative to `simple_fusion_mean`.

Use `mean` for canonical paper and reproduction runs unless you are explicitly
replicating the MIL ablations.

Relevant modules:

- [aggregator.py](aggregator.py): pooled slide-level aggregation helpers
- [mil.py](mil.py): ABMIL and TransMIL bag-level aggregators
- [foundation_models.py](foundation_models.py): pathology foundation-model registry
- [runtime.py](runtime.py): artifact/runtime loader for deployed vision components
