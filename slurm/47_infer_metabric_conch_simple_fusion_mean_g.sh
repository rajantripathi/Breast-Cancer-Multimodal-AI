#!/usr/bin/env bash
#SBATCH --job-name=bcai-metabric-g
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
SEED="${SEED:-42}"
PHASE1_ROOT="${PHASE1_ROOT:-/scratch/u6ef/rajantripathi.u6ef/bcai-phase1-abmil-20260504-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/external/metabric/conch_simple_fusion_mean_genomics_only/seed${SEED}}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -u -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

python -u -m training.tcga_simple_fusion \
  --inference-only \
  --checkpoint "$PHASE1_ROOT/outputs/stage2/conch/simple_fusion_mean/seed${SEED}/model.pt" \
  --reference-crosswalk "$PHASE1_ROOT/data/tcga_crosswalk_conch_pathways.csv" \
  --reference-clinical-csv "$PHASE1_ROOT/data/tcga_brca_clinical.csv" \
  --crosswalk data/external/metabric_crosswalk_genomics_clinical.csv \
  --clinical-csv external/metabric/metadata/clinical.csv \
  --modalities genomics \
  --endpoint 5yr_survival \
  --reference-endpoint pfi \
  --vision-aggregation mean \
  --genomics-aggregation flat \
  --clinical-aggregation flat \
  --seed "$SEED" \
  --device auto \
  --output-dir "$OUTPUT_DIR"
