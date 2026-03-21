#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-pathways
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"

HALLMARK_GMT="${HALLMARK_GMT:-$REPO_DIR/data/reference/h.all.v2024.1.Hs.symbols.gmt}"
if [ ! -f "$HALLMARK_GMT" ]; then
  echo "Missing Hallmark GMT: $HALLMARK_GMT" >&2
  echo "Place the MSigDB Hallmark .gmt file at that path or set HALLMARK_GMT explicitly." >&2
  exit 1
fi

python -u -m data.preprocess.preprocess_tcga_genomics \
  --representation pathways \
  --hallmark-gmt "$HALLMARK_GMT" \
  --output-dir "$PROJECT_ROOT/tcga-brca/genomics_pathways" \
  --matrix-output "$PROJECT_ROOT/tcga-brca/hallmark_expression_matrix.csv"
