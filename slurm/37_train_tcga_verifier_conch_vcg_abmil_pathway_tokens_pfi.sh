#!/usr/bin/env bash
#SBATCH --job-name=bcai-conch-path-tok
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
python -u -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

python -u -m training.tcga_verifier \
  --crosswalk data/tcga_crosswalk_conch_pathways.csv \
  --clinical-csv data/tcga_brca_clinical.csv \
  --modalities vision,clinical,genomics \
  --endpoint pfi \
  --vision-aggregation abmil \
  --genomics-aggregation pathway_tokens \
  --max-vision-instances "${MAX_VISION_INSTANCES:-256}" \
  --device auto \
  --output-dir outputs/tcga_verifier_conch_vcg_abmil_pathway_tokens_pfi_cv
