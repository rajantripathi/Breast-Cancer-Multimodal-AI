#!/bin/bash
#SBATCH --job-name=bcai-mammoclip-train
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/mammoclip_train_%j.out

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

"$VENV_DIR/bin/python" -u -m agents.mammography.training.train_mammoclip_classifier \
  --metadata "$PROJECT_ROOT/data/mammography/vindr-mammo/processed/metadata.csv" \
  --embedding-dir "$PROJECT_ROOT/data/mammography/vindr-mammo/embeddings/mammoclip" \
  --output-dir outputs/mammography/mammoclip \
  --epochs 50 \
  --lr 1e-3 \
  --device auto
