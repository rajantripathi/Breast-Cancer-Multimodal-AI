#!/bin/bash
#SBATCH --job-name=bcai-breast-mv
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/breast_multiview_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
export PROJECT_ROOT
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true
source "$VENV_DIR/bin/activate"

"$VENV_DIR/bin/python" -u -m agents.mammography.training.train_breast_multiview \
  --metadata "$PROJECT_ROOT/data/mammography/vindr-mammo/processed/breast_metadata.csv" \
  --output-dir outputs/mammography/vindr_breast_multiview \
  --backbone-name convnext_base.fb_in22k_ft_in1k \
  --epochs 30 \
  --lr 1e-4 \
  --batch-size 16 \
  --image-size 456 \
  --freeze-epochs 3 \
  --patience 8 \
  --device auto
