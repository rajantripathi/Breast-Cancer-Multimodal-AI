#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-crosswalk
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out

# Submit with: sbatch --dependency=afterok:$ARRAY_JOB_ID slurm/14_build_tcga_crosswalk.sh

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -u -m data.preprocess.build_tcga_crosswalk
