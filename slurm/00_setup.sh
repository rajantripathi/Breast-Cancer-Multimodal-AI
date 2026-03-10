#!/usr/bin/env bash
#SBATCH --job-name=bcai-setup
#SBATCH --account=u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
bash "$REPO_DIR/scripts/isambard/setup_env.sh"
