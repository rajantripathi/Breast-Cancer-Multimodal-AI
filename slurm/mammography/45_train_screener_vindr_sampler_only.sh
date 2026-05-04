#!/bin/bash
# VinDr-only ablation: keep the balanced sampler but remove weighted BCE.

#SBATCH --job-name=bcai-mammo-sonly
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/mammo_sonly_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI}"
MAMMO_DATA_ROOT="${MAMMO_DATA_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mammography/vindr_sampler_only_seed42}"
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
  --loss smoothed_bce \
  --label-smoothing 0.1 \
  --tta none
