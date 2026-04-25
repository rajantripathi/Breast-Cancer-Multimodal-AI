#!/bin/bash
# Stage 1 recovery sweep entrypoint.
# Keeps the canonical mammography benchmark untouched by writing to a separate
# output directory and enabling the stronger class-imbalance defaults.

#SBATCH --job-name=bcai-mammo-recovery
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/mammo_recovery_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI}"
# The processed VinDr-Mammo dataset currently lives in the legacy
# mammography-only repo clone on Isambard. Keep code and data roots separate.
MAMMO_DATA_ROOT="${MAMMO_DATA_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mammography/recovery_seed42}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
SEED="${SEED:-42}"
export PROJECT_ROOT
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true
source "$VENV_DIR/bin/activate"

"$VENV_DIR/bin/python" -u -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

"$VENV_DIR/bin/python" -u -m agents.mammography.training.train_screener \
  --data-dir "$MAMMO_DATA_ROOT/data/mammography/vindr-mammo/processed" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 50 \
  --lr 3e-4 \
  --batch-size 2 \
  --effective-batch-size 8 \
  --image-size "$IMAGE_SIZE" \
  --device auto \
  --seed "$SEED" \
  --balance-sampler \
  --loss weighted_bce \
  --tta none
