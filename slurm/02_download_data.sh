#!/usr/bin/env bash
#SBATCH --job-name=bcai-data
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -m data.download.download_vision
python -m data.download.download_ehr
python -m data.download.download_genomics
python -m data.download.download_literature
