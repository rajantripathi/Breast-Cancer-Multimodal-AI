#!/bin/bash
# Download the exact EMBED DICOM subset referenced by the processed external
# evaluation manifest.

#SBATCH --job-name=bcai-embed-images
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/embed_images_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

EMBED_ROOT="${EMBED_ROOT:-$PROJECT_ROOT/data/mammography/embed}"
RAW_ROOT="${RAW_ROOT:-$EMBED_ROOT/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$EMBED_ROOT/processed}"
MANIFEST_PATH="${MANIFEST_PATH:-$PROCESSED_ROOT/download_manifest.txt}"
LIMIT_IMAGES="${LIMIT_IMAGES:-}"
EMBED_UNSIGNED="${EMBED_UNSIGNED:-1}"

python3 -c "import boto3" >/dev/null 2>&1 || {
  echo "Installing boto3 into $VENV_DIR"
  "$VENV_DIR/bin/pip" install boto3
}

CMD=(
  "$VENV_DIR/bin/python" -u -m data.preprocess.download_embed
  --output-dir "$RAW_ROOT"
  --skip-tables
  --download-images
  --manifest "$MANIFEST_PATH"
)

if [[ -n "$LIMIT_IMAGES" ]]; then
  CMD+=(--limit-images "$LIMIT_IMAGES")
fi
if [[ "$EMBED_UNSIGNED" == "1" ]]; then
  CMD+=(--unsigned)
fi

"${CMD[@]}"
