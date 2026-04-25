#!/bin/bash

# Eval-only GPU job to generate Stage 1 mammography statistical-depth outputs
# for Paper 2 from the final legacy checkpoint.

#SBATCH --job-name=bcai-mammo-stage1-stats
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/mammo_stage1_stats_%j.out

set -euo pipefail

cd /scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI
source scripts/isambard/slurm_env.sh

DATA_DIR="${DATA_DIR:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography/data/mammography/vindr-mammo/processed}"
CHECKPOINT="${CHECKPOINT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI/outputs/mammography/best_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI/reports/paper2/stage1_statistics}"

mkdir -p "$OUTPUT_DIR"

"$VENV_DIR/bin/python" -u -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

"$VENV_DIR/bin/python" -u scripts/paper2_mammo_statistical_depth.py \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device auto \
  --batch-size 8 \
  --image-size 224 \
  --num-workers 4 \
  --bootstrap-iterations 2000
