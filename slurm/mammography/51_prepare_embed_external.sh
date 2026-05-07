#!/bin/bash
# Build EMBED external-evaluation metadata and the exact image download
# manifest for the Stage 1 screener.

#SBATCH --job-name=bcai-embed-prepare
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/embed_prepare_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

EMBED_ROOT="${EMBED_ROOT:-$PROJECT_ROOT/data/mammography/embed}"
RAW_ROOT="${RAW_ROOT:-$EMBED_ROOT/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$EMBED_ROOT/processed}"
EXAM_TYPE="${EXAM_TYPE:-screening}"
LABEL_MODE="${LABEL_MODE:-recall_or_pathology}"
IMAGE_TYPES="${IMAGE_TYPES:-2D,C-view}"
PREFERRED_IMAGE_TYPE="${PREFERRED_IMAGE_TYPE:-2D}"
ALLOW_CVIEW_FALLBACK="${ALLOW_CVIEW_FALLBACK:-1}"
FULL_FIELD_ONLY="${FULL_FIELD_ONLY:-1}"

CMD=(
  "$VENV_DIR/bin/python" -u -m agents.mammography.preprocessing.prepare_embed
  --input-dir "$RAW_ROOT"
  --output-dir "$PROCESSED_ROOT"
  --exam-type "$EXAM_TYPE"
  --label-mode "$LABEL_MODE"
  --image-types "$IMAGE_TYPES"
  --preferred-image-type "$PREFERRED_IMAGE_TYPE"
)

if [[ "$ALLOW_CVIEW_FALLBACK" == "1" ]]; then
  CMD+=(--allow-cview-fallback)
fi
if [[ "$FULL_FIELD_ONLY" == "1" ]]; then
  CMD+=(--full-field-only)
fi

"${CMD[@]}"
