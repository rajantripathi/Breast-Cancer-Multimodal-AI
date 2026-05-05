#!/usr/bin/env bash
#SBATCH --job-name=bcai-virchow2-extract
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/virchow2_extract_%A_%a.out
#SBATCH --array=0-0

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

export PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:-$HOME}/breast-cancer-multimodal-ai}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$PROJECT_ROOT/cache/models}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$MODEL_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_ROOT/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/cache/triton}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true

source "$VENV_DIR/bin/activate"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN must be set for gated Virchow2 access" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SHARDS="${NUM_SHARDS:-32}"
SKIP_PATCH_OUTPUT="${SKIP_PATCH_OUTPUT:-1}"
SLIDE_OUTPUT_DIR="${SLIDE_OUTPUT_DIR:-$REPO_DIR/data/embeddings/virchow2/slides}"
PATCH_OUTPUT_DIR="${PATCH_OUTPUT_DIR:-$REPO_DIR/data/embeddings/virchow2/patches}"
mkdir -p "$SLIDE_OUTPUT_DIR" "$PATCH_OUTPUT_DIR"

"$VENV_DIR/bin/python" -u -c "import timm, huggingface_hub; print('bootstrap_ok')" >/dev/null

if [ -n "${CHUNK_DIR:-}" ]; then
  TILE_LIST="$CHUNK_DIR/chunk_${TASK_ID}.txt"
  if [ ! -f "$TILE_LIST" ]; then
    echo "Missing tile list for task $TASK_ID: $TILE_LIST" >&2
    exit 1
  fi
  "$VENV_DIR/bin/python" -u -m data.preprocess.extract_tcga_features \
    --model virchow2 \
    --batch-size "$BATCH_SIZE" \
    --tile-list "$TILE_LIST" \
    --output-dir "$SLIDE_OUTPUT_DIR" \
    --patch-output-dir "$PATCH_OUTPUT_DIR" \
    ${SKIP_PATCH_OUTPUT:+--skip-patch-output}
else
  "$VENV_DIR/bin/python" -u -m data.preprocess.extract_tcga_features \
    --model virchow2 \
    --batch-size "$BATCH_SIZE" \
    --shard-index "$TASK_ID" \
    --num-shards "$NUM_SHARDS" \
    --output-dir "$SLIDE_OUTPUT_DIR" \
    --patch-output-dir "$PATCH_OUTPUT_DIR" \
    ${SKIP_PATCH_OUTPUT:+--skip-patch-output}
fi
