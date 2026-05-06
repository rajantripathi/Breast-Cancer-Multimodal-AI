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

## CPTAC-BRCA status

The public imaging and RNA sides can be aligned.

- The official TCIA collection page confirms `134` subjects in `CPTAC-BRCA`.
- The TCIA/pathdb cohort-builder CSV is usable even though the older NBIA v1
  probe endpoints returned empty payloads.
- That pathdb manifest contains:
  - `patient_id`
  - `slide_id`
  - `camic_id`
  - direct `wsiimage_url` links to the SVS files
- The public GDC CPTAC breast RNA-seq manifest contains `134` patient IDs.
- The direct patient-ID overlap between TCIA/pathdb and GDC is `122` patients.

Implication:

- `CPTAC-BRCA` is **ready for V+G alignment and slide download**.
- The branch script `data/preprocess/download_cptac_brca.py` now codifies that
  overlap and writes:
  - `external/cptac_brca/metadata/tcia_slide_manifest.csv`
  - `external/cptac_brca/metadata/gdc_rnaseq_manifest.csv`
  - `external/cptac_brca/metadata/alignment_probe.json`

Current limitation:

- I did **not** identify a public survival endpoint for CPTAC-BRCA in the
  sources used here.
- GDC public case fields expose age and pathologic stage, but not a usable
  survival outcome.
- The public `brca_cptac_2020` cBioPortal study exposes receptor/status fields
  but no survival attributes.

So CPTAC currently supports:

- image download
- image/RNA patient alignment
- likely `V+G` or `V` external inference runs

But it does **not yet support a public external C-index calculation** without
an additional outcome source.

## METABRIC status

The public `brca_metabric` cBioPortal study is viable for quantitative external
validation of the non-vision arm.

Verified public attributes include:

- `AGE_AT_DIAGNOSIS`
- `ER_IHC`
- `HER2_SNP6`
- `OS_MONTHS`
- `OS_STATUS`
- `PR_STATUS`
- `TUMOR_STAGE`
- `GRADE`

Verified molecular profiles include:

- `brca_metabric_mrna`
- `brca_metabric_mrna_median_all_sample_Zscores`

The branch script `data/preprocess/download_metabric.py` now downloads the
patient-level clinical metadata from those endpoints into:

- `external/metabric/metadata/clinical_raw.csv`
- `external/metabric/metadata/study.json`
- `external/metabric/metadata/clinical_attributes.json`

## Practical Phase 4 interpretation

- `METABRIC` is the cohort that can produce the first quantitative external
  survival number now.
- `CPTAC-BRCA` is no longer blocked on imaging/RNA alignment, but it is still
  blocked on publicly accessible survival outcomes.
- The next defensible move is:
  1. complete METABRIC `C+G` / `G-only` external evaluation
  2. keep CPTAC as a matched image+RNA external cohort pending outcome labels
