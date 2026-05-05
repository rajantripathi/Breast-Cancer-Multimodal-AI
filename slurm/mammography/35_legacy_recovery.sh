#!/bin/bash
# Phase 1 legacy baseline recovery. Uses the exact f209755 training recipe on
# the current VinDr-Mammo processed dataset to confirm whether the regression is
# in trainer/model code rather than the underlying data split.

#SBATCH --job-name=bcai-mammo-legacy-recovery
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/mammo_legacy_recovery_%j.out

set -euo pipefail

source scripts/isambard/slurm_env.sh
VINDR_PROCESSED_ROOT="${VINDR_PROCESSED_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography/data/mammography/vindr-mammo/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mammography/legacy_recovery}"

"$VENV_DIR/bin/python" -u -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

"$VENV_DIR/bin/python" -u -m agents.mammography.training.train_screener_legacy \
  --data-dir "$VINDR_PROCESSED_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 50 \
  --lr 1e-4 \
  --batch-size 8 \
  --device auto
