#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
export PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:-$HOME}/breast-cancer-multimodal-ai}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$PROJECT_ROOT/cache/models}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$MODEL_CACHE_DIR}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true

source "$VENV_DIR/bin/activate"

