#!/usr/bin/env bash
#SBATCH --job-name=bcai-ui
#SBATCH --account=u6ef
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
echo "UI deployment is intended for local or SSH-tunneled use."
echo "Artifacts are available under outputs/ after evaluation."
