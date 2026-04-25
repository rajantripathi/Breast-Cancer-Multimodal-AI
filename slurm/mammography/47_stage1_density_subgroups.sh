#!/bin/bash

# CPU-only post-processing job to compute Stage 1 density subgroup summaries
# from refreshed metadata and saved Stage 1 predictions.

#SBATCH --job-name=bcai-mammo-density
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/mammo_density_%j.out

set -euo pipefail

cd /scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI
source scripts/isambard/slurm_env.sh

METADATA_CSV="${METADATA_CSV:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography/data/mammography/vindr-mammo/processed/metadata.csv}"
PREDICTIONS_JSON="${PREDICTIONS_JSON:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI/reports/paper2/stage1_statistics/predictions.json}"
OUTPUT_JSON="${OUTPUT_JSON:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI/reports/paper2/stage1_statistics/stage1_density_subgroups.json}"

"$VENV_DIR/bin/python" -u scripts/paper2_mammo_density_subgroups.py \
  --predictions-json "$PREDICTIONS_JSON" \
  --metadata-csv "$METADATA_CSV" \
  --output-json "$OUTPUT_JSON"
