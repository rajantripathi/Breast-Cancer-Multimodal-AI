#!/bin/bash
#SBATCH --job-name=bcai-mammoclip-extract
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/mammoclip_extract_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
MAMMOCLIP_ROOT="${MAMMOCLIP_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Mammo-CLIP}"
export PROJECT_ROOT
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true
source "$VENV_DIR/bin/activate"

MAMMOCLIP_CKPT="$MAMMOCLIP_ROOT/checkpoints/b5-model-best-epoch-7.tar"
VINDR_METADATA="$PROJECT_ROOT/data/mammography/vindr-mammo/processed/metadata.csv"
VINDR_IMAGES="$PROJECT_ROOT/data/mammography/vindr-mammo/processed"
OUTPUT_DIR="$PROJECT_ROOT/data/mammography/vindr-mammo/embeddings/mammoclip"

"$VENV_DIR/bin/python" -u -m agents.mammography.preprocessing.extract_mammoclip_features \
  --checkpoint "$MAMMOCLIP_CKPT" \
  --metadata "$VINDR_METADATA" \
  --image-dir "$VINDR_IMAGES" \
  --output-dir "$OUTPUT_DIR" \
  --mammoclip-root "$MAMMOCLIP_ROOT" \
  --batch-size 16 \
  --device auto
