#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:-$HOME}/breast-cancer-multimodal-ai}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$PROJECT_ROOT/cache/models}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/breast-cancer-multimodal-ai}"

module purge || true
module load cray-python/3.11.7 2>/dev/null || true
module load cudatoolkit/24.11_12.6 2>/dev/null || true

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REPO_DIR/requirements.txt"
python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch torchvision
python -m pip install "timm>=1.0.3" "huggingface-hub>=0.23.0"
python3 -c "import torch; print('torch OK:', torch.__version__); print('cuda built:', torch.backends.cuda.is_built())"
python3 -c "import timm; print('timm OK:', timm.__version__)"
python3 -c "import huggingface_hub; print('hf_hub OK:', huggingface_hub.__version__)"
python3 -c "import openslide; from openslide import OpenSlide; print('openslide import OK')"

mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/processed" "$MODEL_CACHE_DIR" "$RUN_ROOT" "$ARTIFACT_ROOT"
mkdir -p "$DATA_ROOT/splits" "$PROJECT_ROOT/cache/huggingface"
echo "Remote environment ready in $VENV_DIR"
