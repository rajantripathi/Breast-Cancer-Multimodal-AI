#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-eval
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -u -m evaluation.evaluate \
  --experiment-dir outputs/tcga_verifier \
  --clinical-csv data/tcga_brca_clinical.csv \
  --output-dir reports/tcga_evaluation
