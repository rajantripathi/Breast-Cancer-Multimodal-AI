#!/usr/bin/env bash
#SBATCH --job-name=bcai-external-summary
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
REPO_DIR="${REPO_DIR:-$PWD}"
PHASE1_ROOT="${PHASE1_ROOT:-/scratch/u6ef/rajantripathi.u6ef/bcai-phase1-abmil-20260504-1}"

cd "$REPO_DIR"
mkdir -p reports/paper2 logs
source "$REPO_DIR/scripts/isambard/slurm_env.sh"

python -u scripts/external_validation_summary.py \
  --internal-root "$PHASE1_ROOT/outputs/stage2/conch/simple_fusion_mean" \
  --metabric-cg-root outputs/external/metabric/conch_simple_fusion_mean_clinical_genomics \
  --metabric-g-root outputs/external/metabric/conch_simple_fusion_mean_genomics_only \
  --metabric-clinical external/metabric/metadata/clinical.csv \
  --cptac-alignment-probe external/cptac_brca/metadata/alignment_probe.json \
  --output-json reports/paper2/external_validation_summary.json \
  --output-markdown reports/paper2/external_validation_writeup.md
