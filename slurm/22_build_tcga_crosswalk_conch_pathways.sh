#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-xwalk-conch
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"

python -u -m data.preprocess.build_tcga_crosswalk \
  --vision-root "$PROJECT_ROOT/tcga-brca/embeddings/conch" \
  --genomics-root "$PROJECT_ROOT/tcga-brca/genomics_pathways" \
  --output-csv "$REPO_DIR/data/tcga_crosswalk_conch_pathways.csv" \
  --report-path "$REPO_DIR/reports/tcga_alignment_report_conch_pathways.txt"
