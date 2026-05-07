#!/bin/bash
# Download the EMBED table bundle onto Isambard.
# This is intentionally tables-first because EMBED is large; image transfer
# should follow the processed cohort manifest created by prepare_embed.py.

#SBATCH --job-name=bcai-embed-tables
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/embed_tables_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

EMBED_ROOT="${EMBED_ROOT:-$PROJECT_ROOT/data/mammography/embed}"

python3 -c "import boto3" >/dev/null 2>&1 || {
  echo "Installing boto3 into $VENV_DIR"
  "$VENV_DIR/bin/pip" install boto3
}

"$VENV_DIR/bin/python" -u -m data.preprocess.download_embed \
  --output-dir "$EMBED_ROOT/raw"
