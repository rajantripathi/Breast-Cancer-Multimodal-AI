#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-tile
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python3 -c "import openslide; from openslide import OpenSlide; print('OpenSlide OK')" >/dev/null 2>&1 || {
  echo "FATAL: OpenSlide runtime unavailable in the project venv. Rerun slurm/00_setup.sh after installing openslide-bin." >&2
  exit 1
}
python -m data.preprocess.tile_tcga
