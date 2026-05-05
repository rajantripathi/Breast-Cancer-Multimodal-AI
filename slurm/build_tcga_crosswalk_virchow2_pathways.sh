#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-xwalk-vir2
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
  --vision-root "$REPO_DIR/data/embeddings/virchow2/slides" \
  --patch-vision-root "$REPO_DIR/data/embeddings/virchow2/patches" \
  --genomics-root "$PROJECT_ROOT/tcga-brca/genomics_pathways" \
  --output-csv "$REPO_DIR/data/tcga_crosswalk_virchow2_pathways.csv" \
  --report-path "$REPO_DIR/reports/tcga_alignment_report_virchow2_pathways.txt"
