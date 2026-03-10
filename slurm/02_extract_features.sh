#!/usr/bin/env bash
#SBATCH --job-name=bcai-vfeat
#SBATCH --account=brics.u6ef
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

for MODEL in uni2 conch virchow2 ctranspath; do
  echo "=== Extracting features with ${MODEL} ==="
  python -m agents.vision.extract_features \
    --config config/isambard.yaml \
    --model "${MODEL}" \
    --data-dir "${DATA_ROOT:-$REPO_DIR/data}" \
    --manifest "${DATA_ROOT:-$REPO_DIR/data}/processed/vision/feature_manifest.jsonl" \
    --output-dir "$REPO_DIR/outputs/vision/features/${MODEL}" \
    --batch-size 256 \
    --num-workers 8 \
    "${SMOKE_FLAG[@]}"
done
