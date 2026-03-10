#!/usr/bin/env bash
#SBATCH --job-name=bcai-ehr
#SBATCH --account=u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
SMOKE_FLAG=()
[ "${SMOKE_TEST:-0}" = "1" ] && SMOKE_FLAG+=(--smoke-test)
python -m training.ehr_trainer --config experiments/configs/ehr_metabric.yaml --device cpu "${SMOKE_FLAG[@]}"
