#!/usr/bin/env bash
#SBATCH --job-name=bcai-prep
#SBATCH --account=u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -m data.preprocess.preprocess_vision
python -m data.preprocess.preprocess_ehr
python -m data.preprocess.preprocess_genomics
python -m data.preprocess.preprocess_literature
python -m data.splits.create_splits
