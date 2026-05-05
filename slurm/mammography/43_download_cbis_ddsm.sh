#!/bin/bash
# Download CBIS-DDSM from IDC/TCIA onto Isambard.
# This fetches the MG image series plus the official case-description CSVs.

#SBATCH --job-name=bcai-cbis-download
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cbis_download_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

CBIS_ROOT="${PROJECT_ROOT}/data/mammography/cbis-ddsm/raw"
IDC_ROOT="${CBIS_ROOT}/idc"
METADATA_ROOT="${CBIS_ROOT}/metadata"

mkdir -p "$IDC_ROOT" "$METADATA_ROOT"

echo "Downloading CBIS-DDSM to $CBIS_ROOT"
echo "Image source: IDC mirror of TCIA collection cbis_ddsm"
echo "Metadata source: official TCIA case-description CSVs"

python3 -c "import idc_index" >/dev/null 2>&1 || {
  echo "Installing idc-index into $VENV_DIR"
  "$VENV_DIR/bin/pip" install idc-index
}

CBIS_IDC_ROOT="$IDC_ROOT" python3 - <<'PY'
import os
from idc_index import index

download_dir = os.environ["CBIS_IDC_ROOT"]
client = index.IDCClient()
client.download_collection(
    "cbis_ddsm",
    download_dir,
    dry_run=False,
    quiet=False,
    show_progress_bar=False,
    source_bucket_location="aws",
)
PY

wget -q -O "$METADATA_ROOT/mass_case_description_train_set.csv" \
  "https://www.cancerimagingarchive.net/wp-content/uploads/mass_case_description_train_set.csv"
wget -q -O "$METADATA_ROOT/calc_case_description_train_set.csv" \
  "https://www.cancerimagingarchive.net/wp-content/uploads/calc_case_description_train_set.csv"
wget -q -O "$METADATA_ROOT/mass_case_description_test_set.csv" \
  "https://www.cancerimagingarchive.net/wp-content/uploads/mass_case_description_test_set.csv"
wget -q -O "$METADATA_ROOT/calc_case_description_test_set.csv" \
  "https://www.cancerimagingarchive.net/wp-content/uploads/calc_case_description_test_set.csv"
wget -q -O "$CBIS_ROOT/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia" \
  "https://www.cancerimagingarchive.net/wp-content/uploads/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia"

echo "CBIS-DDSM download complete"
find "$METADATA_ROOT" -maxdepth 1 -type f -name '*.csv' | sort
