#!/bin/bash
# Evaluate a trained non-legacy screener checkpoint on the prepared EMBED
# external cohort.

#SBATCH --job-name=bcai-embed-ext-eval
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/embed_external_eval_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the trained non-legacy checkpoint}"
EMBED_ROOT="${EMBED_ROOT:-$PROJECT_ROOT/data/mammography/embed}"
DATA_DIR="${DATA_DIR:-$EMBED_ROOT/processed}"
METADATA_CSV="${METADATA_CSV:-$DATA_DIR/metadata.csv}"
RAW_DIR="${RAW_DIR:-$EMBED_ROOT/raw/images}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mammography/external_eval_embed}"
HARMONIZATION_STATS_JSON="${HARMONIZATION_STATS_JSON:-}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

CMD=(
  "$VENV_DIR/bin/python" -u -m agents.mammography.evaluation.evaluate_screener
  --model-path "$MODEL_PATH"
  --model-type standard
  --data-dir "$DATA_DIR"
  --metadata-csv "$METADATA_CSV"
  --raw-dir "$RAW_DIR"
  --eval-split external
  --image-size "$IMAGE_SIZE"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$HARMONIZATION_STATS_JSON" ]]; then
  CMD+=(--harmonization-stats-json "$HARMONIZATION_STATS_JSON")
fi

"${CMD[@]}"
