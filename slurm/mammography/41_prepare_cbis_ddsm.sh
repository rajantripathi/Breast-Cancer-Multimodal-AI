#!/bin/bash
# Prepare CBIS-DDSM as a train-only auxiliary mammography source.
# Assumes the raw dataset and metadata CSVs have already been staged locally.

#SBATCH --job-name=bcai-cbis-prep
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cbis_prepare_%j.out

set -euo pipefail

source scripts/isambard/slurm_env.sh

CBIS_ROOT="${PROJECT_ROOT}/data/mammography/cbis-ddsm/raw"
CBIS_OUT="${PROJECT_ROOT}/data/mammography/cbis-ddsm/processed"
METADATA_ROOT="${CBIS_ROOT}/metadata"

"$VENV_DIR/bin/python" -u -m agents.mammography.preprocessing.prepare_cbis_ddsm \
  --input-dir "$CBIS_ROOT" \
  --output-dir "$CBIS_OUT" \
  --metadata-csv "$METADATA_ROOT/calc_case_description_test_set.csv" \
  --metadata-csv "$METADATA_ROOT/calc_case_description_train_set.csv" \
  --metadata-csv "$METADATA_ROOT/mass_case_description_test_set.csv" \
  --metadata-csv "$METADATA_ROOT/mass_case_description_train_set.csv" \
  --image-size 1536
