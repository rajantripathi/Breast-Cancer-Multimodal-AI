#!/bin/bash
#SBATCH --job-name=bcai-vindr-breast-meta
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/vindr_breast_meta_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
export PROJECT_ROOT
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
source "$VENV_DIR/bin/activate"

"$VENV_DIR/bin/python" -u -m agents.mammography.preprocessing.build_vindr_breast_metadata \
  --metadata "$PROJECT_ROOT/data/mammography/vindr-mammo/processed/metadata.csv" \
  --output "$PROJECT_ROOT/data/mammography/vindr-mammo/processed/breast_metadata.csv" \
  --require-both-views
