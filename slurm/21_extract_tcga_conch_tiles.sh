#!/usr/bin/env bash
#SBATCH --job-name=bcai-conch-extract
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/conch_extract_%A_%a.out
#SBATCH --array=0-0

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

if [ -z "${CONCH_HF_TOKEN:-}" ]; then
  echo "CONCH_HF_TOKEN must be set in the environment or sbatch --export" >&2
  exit 1
fi

CHUNK_DIR="${CHUNK_DIR:?Set CHUNK_DIR to the directory containing chunk_*.txt files}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TILE_LIST="$CHUNK_DIR/chunk_${TASK_ID}.txt"
if [ ! -f "$TILE_LIST" ]; then
  echo "Missing tile list for task $TASK_ID: $TILE_LIST" >&2
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-1}"

python -u -m data.preprocess.extract_tcga_features \
  --model conch \
  --batch-size "$BATCH_SIZE" \
  --tile-list "$TILE_LIST" \
  --output-dir "$PROJECT_ROOT/tcga-brca/embeddings/conch" \
  --patch-output-dir "$PROJECT_ROOT/tcga-brca/patch_embeddings/conch"
