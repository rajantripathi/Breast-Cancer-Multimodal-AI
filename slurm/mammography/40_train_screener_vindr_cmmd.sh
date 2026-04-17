#!/bin/bash
# Train the recovered legacy screener on VinDr with CMMD as a train-only
# auxiliary source. Validation and test remain VinDr-only.

#SBATCH --job-name=bcai-mammo-vcmmd
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/mammo_vcmmd_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

VINDR_PROCESSED_ROOT="${VINDR_PROCESSED_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography/data/mammography/vindr-mammo/processed}"
CMMD_PROCESSED_ROOT="${CMMD_PROCESSED_ROOT:-$PROJECT_ROOT/data/mammography/cmmd/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mammography/vindr_cmmd_legacy_seed42}"
SEED="${SEED:-42}"

"$VENV_DIR/bin/python" -u -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

"$VENV_DIR/bin/python" -u -m agents.mammography.training.train_screener_multisource \
  --data-dir "$VINDR_PROCESSED_ROOT" \
  --metadata-csv "$VINDR_PROCESSED_ROOT/metadata.csv" \
  --aux-metadata-csv "$CMMD_PROCESSED_ROOT/metadata.csv" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 50 \
  --lr 1e-4 \
  --batch-size 8 \
  --device auto \
  --seed "$SEED" \
  --image-size 224
