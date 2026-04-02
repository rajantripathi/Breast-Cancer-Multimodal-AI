#!/bin/bash
#SBATCH --job-name=bcai-breast-mc
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/breast_mammoclip_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
export PROJECT_ROOT
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

source "$VENV_DIR/bin/activate"

"$VENV_DIR/bin/python" -u -m agents.mammography.training.train_breast_mammoclip_multiview \
  --metadata "$PROJECT_ROOT/data/mammography/vindr-mammo/processed/breast_metadata.csv" \
  --embedding-dir "$PROJECT_ROOT/data/mammography/vindr-mammo/embeddings/mammoclip" \
  --output-dir outputs/mammography/mammoclip_breast_multiview \
  --epochs 30 \
  --lr 1e-3 \
  --batch-size 64 \
  --patience 8 \
  --device cpu
