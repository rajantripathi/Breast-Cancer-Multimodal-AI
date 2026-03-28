#!/bin/bash
#SBATCH --job-name=bcai-mammo-train
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/mammo_train_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

# Verify GPU
python -u -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"

python -u -m agents.mammography.training.train_screener \
  --data-dir "$PROJECT_ROOT/data/mammography/vindr-mammo/processed" \
  --output-dir outputs/mammography \
  --epochs 50 \
  --lr 1e-4 \
  --batch-size 8 \
  --device auto
