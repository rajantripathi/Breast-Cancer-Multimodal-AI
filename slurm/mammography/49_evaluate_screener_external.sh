#!/bin/bash
# Evaluate a non-legacy Stage 1 screener checkpoint on a metadata-compatible
# external cohort using the train-fit harmonization statistics from training.

#SBATCH --job-name=bcai-mammo-ext-eval
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/mammo_external_eval_%j.out

set -euo pipefail

source scripts/isambard/slurm_env.sh

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the trained non-legacy checkpoint}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the processed external cohort directory}"
METADATA_CSV="${METADATA_CSV:-$DATA_DIR/metadata.csv}"
RAW_DIR="${RAW_DIR:-$DATA_DIR/../raw}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mammography/external_eval}"
HARMONIZATION_STATS_JSON="${HARMONIZATION_STATS_JSON:-}"
EVAL_SPLIT="${EVAL_SPLIT:-external}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

CMD=(
  "$VENV_DIR/bin/python" -u -m agents.mammography.evaluation.evaluate_screener
  --model-path "$MODEL_PATH"
  --model-type standard
  --data-dir "$DATA_DIR"
  --metadata-csv "$METADATA_CSV"
  --raw-dir "$RAW_DIR"
  --eval-split "$EVAL_SPLIT"
  --image-size "$IMAGE_SIZE"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$HARMONIZATION_STATS_JSON" ]]; then
  CMD+=(--harmonization-stats-json "$HARMONIZATION_STATS_JSON")
fi

"${CMD[@]}"
