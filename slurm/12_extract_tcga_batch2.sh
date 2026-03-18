#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-extract-b2
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_batch2_%a_%j.out
#SBATCH --array=0-31

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
python -u -m data.preprocess.extract_tcga_features \
  --model uni2 \
  --batch-size 2 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 32
