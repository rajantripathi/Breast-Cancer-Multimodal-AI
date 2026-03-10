#!/usr/bin/env bash
#SBATCH --job-name=bcai-vfeat
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"
set -a
[ -f "$REPO_DIR/.env" ] && source "$REPO_DIR/.env"
set +a
source "$REPO_DIR/scripts/isambard/slurm_env.sh"
export HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
  echo "WARNING: HF_TOKEN is unset. UNI2, CONCH, and Virchow2 may fail; CTransPath can still work."
fi
python3 -c "import timm; print('timm version:', timm.__version__)"
python3 -c "import huggingface_hub; print('huggingface_hub version:', huggingface_hub.__version__)"
SMOKE_FLAG=()
[ "${SMOKE_TEST:-0}" = "1" ] && SMOKE_FLAG+=(--smoke-test)
python -m agents.vision.extract_features \
  --config config/isambard.yaml \
  --model "${VISION_MODEL_KEY:-uni2}" \
  --data-dir "${DATA_ROOT:-$REPO_DIR/data}" \
  --manifest "${DATA_ROOT:-$REPO_DIR/data}/processed/vision/feature_manifest.jsonl" \
  --output-dir "$REPO_DIR/outputs/vision/features/${VISION_MODEL_KEY:-uni2}" \
  --batch-size "${VISION_BATCH_SIZE:-16}" \
  --num-workers "${VISION_NUM_WORKERS:-4}" \
  "${SMOKE_FLAG[@]}"
