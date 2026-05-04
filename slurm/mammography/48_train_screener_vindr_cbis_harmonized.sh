#!/bin/bash
# Diagnostic ablation: train the non-legacy screener on VinDr with CBIS-DDSM
# as a train-only auxiliary source, using train-fit source-aware percentile
# harmonization to reduce acquisition-domain contrast mismatch.

#SBATCH --job-name=bcai-mammo-vcaux-h
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/mammo_vcaux_h_%j.out

set -euo pipefail

VINDR_PROCESSED_ROOT="${VINDR_PROCESSED_ROOT:-/scratch/u6ef/rajantripathi.u6ef/Breast-Cancer-Multimodal-AI-mammography/data/mammography/vindr-mammo/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mammography/vindr_cbis_harmonized_seed42}"
SEED="${SEED:-42}"
source scripts/isambard/slurm_env.sh
CBIS_PROCESSED_ROOT="${CBIS_PROCESSED_ROOT:-$PROJECT_ROOT/data/mammography/cbis-ddsm/processed}"

"$VENV_DIR/bin/python" -u -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

"$VENV_DIR/bin/python" -u -m agents.mammography.training.train_screener \
  --data-dir "$VINDR_PROCESSED_ROOT" \
  --metadata-csv "$VINDR_PROCESSED_ROOT/metadata.csv" \
  --aux-metadata-csv "$CBIS_PROCESSED_ROOT/metadata.csv" \
  --source-weight vindr=1.0 \
  --source-weight cbis_ddsm=1.0 \
  --harmonization-method source_percentile \
  --harmonization-lower-quantile 0.01 \
  --harmonization-upper-quantile 0.99 \
  --harmonization-max-images-per-source 256 \
  --output-dir "$OUTPUT_DIR" \
  --epochs 35 \
  --lr 3e-4 \
  --batch-size 2 \
  --effective-batch-size 8 \
  --image-size 224 \
  --device auto \
  --seed "$SEED" \
  --loss smoothed_bce \
  --label-smoothing 0.1 \
  --rotation-degrees 10 \
  --crop-scale-min 0.9 \
  --brightness-jitter 0.15 \
  --contrast-jitter 0.15 \
  --tta none
