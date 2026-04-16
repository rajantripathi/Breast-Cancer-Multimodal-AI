#!/bin/bash
# Preprocess CMMD for auxiliary VinDr+CMMD training.

#SBATCH --job-name=bcai-cmmd-prep
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cmmd_prepare_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

CMMD_ROOT="${PROJECT_ROOT}/data/mammography/cmmd/raw"
CMMD_OUT="${PROJECT_ROOT}/data/mammography/cmmd/processed"
CMMD_XLSX="${CMMD_ROOT}/metadata/CMMD_clinicaldata_revision.xlsx"

"$VENV_DIR/bin/python" -u -m agents.mammography.preprocessing.prepare_cmmd \
  --input-dir "$CMMD_ROOT" \
  --output-dir "$CMMD_OUT" \
  --clinical-xlsx "$CMMD_XLSX" \
  --image-size 224
