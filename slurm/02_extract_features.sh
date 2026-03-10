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

mkdir -p "$REPO_DIR/outputs/vision"

declare -A MODEL_STATUS=()
succeeded=0
failed=0

for MODEL in ctranspath uni2 conch virchow2; do
  echo "=== START ${MODEL} ==="
  set +e
  python -m agents.vision.extract_features \
    --config config/isambard.yaml \
    --model "${MODEL}" \
    --data-dir "${DATA_ROOT:-$REPO_DIR/data}" \
    --manifest "${DATA_ROOT:-$REPO_DIR/data}/processed/vision/feature_manifest.jsonl" \
    --output-dir "$REPO_DIR/outputs/vision/features/${MODEL}" \
    --batch-size 256 \
    --num-workers 8 \
    "${SMOKE_FLAG[@]}"
  exit_code=$?
  set -e

  if [ "$exit_code" -eq 0 ]; then
    MODEL_STATUS["$MODEL"]="success"
    succeeded=$((succeeded + 1))
    echo "=== SUCCESS ${MODEL} ==="
  else
    if grep -q "GatedRepoError\|gated repo\|restricted and you are not in the authorized list\|ask for access" "logs/${SLURM_JOB_ID}.out"; then
      MODEL_STATUS["$MODEL"]="failed_gated_access"
    else
      MODEL_STATUS["$MODEL"]="failed"
    fi
    failed=$((failed + 1))
    echo "=== FAILED ${MODEL} (${MODEL_STATUS[$MODEL]}) ==="
  fi
done

status_file="$REPO_DIR/outputs/vision/extraction_status.json"
printf '{' > "$status_file"
for model_name in ctranspath uni2 conch virchow2; do
  printf '"%s": "%s", ' "$model_name" "${MODEL_STATUS[$model_name]:-not_run}" >> "$status_file"
done
printf '"succeeded": %s, "failed": %s}\n' "$succeeded" "$failed" >> "$status_file"

echo "Extraction summary: succeeded=$succeeded failed=$failed status_file=$status_file"
[ "$succeeded" -gt 0 ] && exit 0
exit 1
