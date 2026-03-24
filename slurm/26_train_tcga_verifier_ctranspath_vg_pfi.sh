#!/usr/bin/env bash
#SBATCH --job-name=bcai-ctr-vg-pfi
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -u -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

python -u -m training.tcga_verifier \
  --crosswalk data/tcga_crosswalk_ctranspath_pathways.csv \
  --clinical-csv data/tcga_brca_clinical.csv \
  --modalities vision,genomics \
  --endpoint pfi \
  --device auto \
  --output-dir outputs/tcga_verifier_ctranspath_vg_pfi_cv
