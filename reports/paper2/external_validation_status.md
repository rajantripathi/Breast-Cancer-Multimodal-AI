# External Validation Status

## Repo readiness

- `agents/vision/foundation_models.py` already supports `CONCH`.
- `training/tcga_simple_fusion.py` now supports `--inference-only` with:
  - checkpoint loading
  - reconstruction of the last-fold TCGA clinical normalization
  - external crosswalk inference
  - `predictions.csv`, `predictions.json`, `summary.json`, and `calibration.json`
- `training/tcga_verifier.py` now supports the same `--inference-only` path.
- `evaluation/statistics.py` already provides:
  - bootstrap confidence intervals
  - paired bootstrap
  - DeLong ROC comparison
  - calibration and Brier utilities

## CPTAC-BRCA blocker

The genomics side is reachable through `GDC`:

- CPTAC breast RNA-seq `Gene Expression Quantification` files are available.
- The current probe found `135` RNA-seq files under `CPTAC-2`.
- GDC submitter IDs use the expected CPTAC subject form, e.g. `05BR001`.

The imaging side is not yet consumable programmatically from the current
Isambard environment:

- The official collection page confirms `134` subjects in `CPTAC-BRCA`.
- The public NBIA v1 probe endpoints for `CPTAC-BRCA` returned empty payloads
  for `getPatient` and `getSeries` from Isambard.
- That means the patient-level TCIA imaging manifest needed for a deterministic
  subject join is not available through the current direct API path.

## Stop-rule outcome

Phase 4 is blocked before WSI download and before inference.

Reason:

- patient-level TCIA to GDC alignment count cannot be computed reliably from
  the current environment because the TCIA side is not exposing a usable
  machine-readable patient manifest for `CPTAC-BRCA`.

The branch contains a codified probe at:

- `data/preprocess/download_cptac_brca.py`

That script writes:

- `external/cptac_brca/metadata/cptac_collection_probe.json`
- `external/cptac_brca/metadata/tcia_nbia_probe.json`
- `external/cptac_brca/metadata/gdc_rnaseq_manifest.json`
- `external/cptac_brca/metadata/alignment_probe.json`
