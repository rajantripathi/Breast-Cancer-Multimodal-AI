#!/usr/bin/env bash
#SBATCH --job-name=bcai-tcga-ver-pw
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
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
python -u -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -u -m training.verifier_trainer \
  --crosswalk data/tcga_crosswalk_pathways.csv \
  --vision-dir "$PROJECT_ROOT/tcga-brca/embeddings/uni2" \
  --genomics-dir "$PROJECT_ROOT/tcga-brca/genomics_pathways" \
  --clinical-csv data/tcga_brca_clinical.csv \
  --modalities vision,clinical,genomics \
  --epochs 100 \
  --lr 1e-4 \
  --patience 20 \
  --device auto \
  --output-dir outputs/tcga_verifier_pathways
