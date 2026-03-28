#!/bin/bash
#SBATCH --job-name=bcai-vindr-download
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vindr_download_%j.out

# Download VinDr-Mammo from PhysioNet
# REQUIRES: PhysioNet credentials configured
# See: https://physionet.org/content/vindr-mammo/1.0.0/

set -euo pipefail
source scripts/isambard/slurm_env.sh

OUTPUT_DIR="$PROJECT_ROOT/data/mammography/vindr-mammo/raw"
mkdir -p "$OUTPUT_DIR"

echo "Downloading VinDr-Mammo to $OUTPUT_DIR"
echo "NOTE: You must have PhysioNet credentials configured"
echo "Run: wget -r -N -c -np --user YOUR_USER --ask-password https://physionet.org/files/vindr-mammo/1.0.0/"

# Uncomment after configuring credentials:
# cd "$OUTPUT_DIR"
# wget -r -N -c -np --user YOUR_USER --ask-password \
#   https://physionet.org/files/vindr-mammo/1.0.0/

echo "Download script ready. Configure PhysioNet credentials first."
