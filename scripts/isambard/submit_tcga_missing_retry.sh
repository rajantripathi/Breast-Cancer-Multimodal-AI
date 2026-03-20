#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"

MODEL_KEY="${MODEL_KEY:-uni2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
LIST_PATH="${LIST_PATH:-$REPO_DIR/reports/tcga_missing_tiles_${MODEL_KEY}.txt}"

python -u -m data.preprocess.build_missing_tcga_tile_list --model "$MODEL_KEY" --output "$LIST_PATH"

MISSING_COUNT="$(grep -cve '^[[:space:]]*$' "$LIST_PATH" || true)"
if [ "$MISSING_COUNT" -eq 0 ]; then
  echo "No missing tiles remain for model ${MODEL_KEY}" >&2
  exit 0
fi

ARRAY_MAX=$(( (MISSING_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE - 1 ))
echo "Submitting missing-tile retry: missing_count=${MISSING_COUNT} chunk_size=${CHUNK_SIZE} array=0-${ARRAY_MAX} batch_size=${BATCH_SIZE}" >&2

sbatch \
  --array="0-${ARRAY_MAX}" \
  --export="ALL,REPO_DIR=${REPO_DIR},TILE_LIST=${LIST_PATH},CHUNK_SIZE=${CHUNK_SIZE},BATCH_SIZE=${BATCH_SIZE},MODEL_KEY=${MODEL_KEY}" \
  slurm/17_retry_tcga_missing_tiles.sh
