#!/bin/bash
# Exact provenance recovery run for the retained 224px ConvNeXt baseline.

#SBATCH --job-name=bcai-mammo-l224
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/mammo_l224_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI}"
MAMMO_DATA_ROOT="${MAMMO_DATA_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mammography/repro_legacy224_seed42}"
SEED="${SEED:-42}"
export PROJECT_ROOT
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true
source "$VENV_DIR/bin/activate"

"$VENV_DIR/bin/python" -u -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

"$VENV_DIR/bin/python" -u -m agents.mammography.training.train_screener_legacy \
  --data-dir "$MAMMO_DATA_ROOT/data/mammography/vindr-mammo/processed" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 50 \
  --lr 1e-4 \
  --batch-size 8 \
  --image-size 224 \
  --device auto \
  --seed "$SEED"
