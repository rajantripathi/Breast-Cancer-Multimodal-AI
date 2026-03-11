#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-download
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -m data.download.download_tcga_brca --max-slides 50 --max-samples 50
# Remove --max-slides/--max-samples for full download.
