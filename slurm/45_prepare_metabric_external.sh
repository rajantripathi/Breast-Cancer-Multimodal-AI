#!/usr/bin/env bash
#SBATCH --job-name=bcai-metabric-prep
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
mkdir -p logs data/external external/metabric
source "$REPO_DIR/scripts/isambard/slurm_env.sh"

python -u -m data.preprocess.download_metabric \
  --output-dir external/metabric \
  --crosswalk-output data/external/metabric_crosswalk_genomics_clinical.csv \
  --hallmark-gmt data/reference/h.all.v2026.1.Hs.symbols.gmt \
  --molecular-profile-id "${METABRIC_MOLECULAR_PROFILE:-brca_metabric_mrna}" \
  --sample-list-id "${METABRIC_SAMPLE_LIST:-brca_metabric_all}" \
  --expression-batch-size "${METABRIC_EXPR_BATCH_SIZE:-32}"
