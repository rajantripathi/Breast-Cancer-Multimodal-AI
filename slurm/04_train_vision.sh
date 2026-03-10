#!/usr/bin/env bash
#SBATCH --job-name=bcai-vision
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
SMOKE_FLAG=()
[ "${SMOKE_TEST:-0}" = "1" ] && SMOKE_FLAG+=(--smoke-test)
python -m training.vision_trainer --config experiments/configs/vision_mias.yaml --device cuda "${SMOKE_FLAG[@]}"
