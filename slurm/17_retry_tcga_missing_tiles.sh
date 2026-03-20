#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-missing
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_missing_%A_%a.out
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

export HF_TOKEN="${HF_TOKEN:-}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

TILE_LIST="${TILE_LIST:?Set TILE_LIST to a newline-delimited list of tile paths}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_KEY="${MODEL_KEY:-uni2}"

case "$CHUNK_SIZE" in
  ''|*[!0-9]*)
    echo "Invalid CHUNK_SIZE=$CHUNK_SIZE" >&2
    exit 1
    ;;
esac

if [ "$CHUNK_SIZE" -lt 1 ]; then
  echo "CHUNK_SIZE must be >= 1" >&2
  exit 1
fi

LIST_PATH="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$TILE_LIST")"
if [ ! -f "$LIST_PATH" ]; then
  echo "Tile list not found: $LIST_PATH" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_LINE=$((TASK_ID * CHUNK_SIZE + 1))
END_LINE=$((START_LINE + CHUNK_SIZE - 1))
TMP_LIST="${TMPDIR:-/tmp}/tcga_missing_${SLURM_JOB_ID}_${TASK_ID}.txt"

sed -n "${START_LINE},${END_LINE}p" "$LIST_PATH" > "$TMP_LIST"
if [ ! -s "$TMP_LIST" ]; then
  echo "No tile paths assigned for task ${TASK_ID} from ${LIST_PATH}" >&2
  exit 0
fi

echo "Processing missing-tile chunk task=${TASK_ID} start_line=${START_LINE} end_line=${END_LINE} batch_size=${BATCH_SIZE}" >&2

python -u -m data.preprocess.extract_tcga_features \
  --model "$MODEL_KEY" \
  --batch-size "$BATCH_SIZE" \
  --tile-list "$TMP_LIST"
