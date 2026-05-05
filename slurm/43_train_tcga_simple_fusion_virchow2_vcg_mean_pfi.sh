#!/usr/bin/env bash
#SBATCH --job-name=bcai-sf-vir2-mean
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2/virchow2/simple_fusion_mean/seed${SEED}}"
cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -u -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

python -u -m training.tcga_simple_fusion \
  --crosswalk data/tcga_crosswalk_virchow2_pathways.csv \
  --clinical-csv data/tcga_brca_clinical.csv \
  --modalities vision,clinical,genomics \
  --endpoint pfi \
  --vision-aggregation mean \
  --genomics-aggregation flat \
  --clinical-aggregation flat \
  --max-vision-instances "${MAX_VISION_INSTANCES:-256}" \
  --seed "$SEED" \
  --device auto \
  --output-dir "$OUTPUT_DIR"
