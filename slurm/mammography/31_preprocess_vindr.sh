#!/bin/bash
#SBATCH --job-name=bcai-vindr-preprocess
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vindr_preprocess_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

python -u -m agents.mammography.preprocessing.prepare_vindr \
  --input-dir "$PROJECT_ROOT/data/mammography/vindr-mammo/raw" \
  --output-dir "$PROJECT_ROOT/data/mammography/vindr-mammo/processed" \
  --image-size 1024
