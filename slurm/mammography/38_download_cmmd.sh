#!/bin/bash
# Download CMMD from IDC/TCIA onto Isambard.

#SBATCH --job-name=bcai-cmmd-download
#SBATCH --partition=workq
#SBATCH --account=brics.u6ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cmmd_download_%j.out

set -euo pipefail
source scripts/isambard/slurm_env.sh

CMMD_ROOT="${PROJECT_ROOT}/data/mammography/cmmd/raw"
IDC_ROOT="${CMMD_ROOT}/idc"
METADATA_ROOT="${CMMD_ROOT}/metadata"

mkdir -p "$IDC_ROOT" "$METADATA_ROOT"

echo "Downloading CMMD to $CMMD_ROOT"
echo "Image source: IDC mirror of TCIA collection cmmd"
echo "Clinical source: CMMD collection page XLSX link"

python3 -c "import idc_index" >/dev/null 2>&1 || {
  echo "Installing idc-index into $VENV_DIR"
  "$VENV_DIR/bin/pip" install idc-index
}

CMMD_IDC_ROOT="$IDC_ROOT" python3 - <<'PY'
import os
from idc_index import index

download_dir = os.environ["CMMD_IDC_ROOT"]
client = index.IDCClient()
client.download_collection(
    "cmmd",
    download_dir,
    dry_run=False,
    quiet=False,
    show_progress_bar=False,
    source_bucket_location="aws",
)
PY

CMMD_METADATA_ROOT="$METADATA_ROOT" python3 - <<'PY'
import re
import urllib.request
import os
from pathlib import Path

page_url = "https://www.cancerimagingarchive.net/collection/cmmd/"
html = urllib.request.urlopen(page_url).read().decode("utf-8", errors="ignore")
match = re.search(r'https://[^"\']+CMMD[^"\']+\.xlsx', html, flags=re.IGNORECASE)
if match is None:
    match = re.search(r'https://[^"\']+clinicaldata[^"\']+\.xlsx', html, flags=re.IGNORECASE)
if match is None:
    raise SystemExit("Could not locate CMMD clinical XLSX link on the collection page")
target = Path(os.environ["CMMD_METADATA_ROOT"]) / "CMMD_clinicaldata_revision.xlsx"
urllib.request.urlretrieve(match.group(0), target)
print(f"Downloaded clinical XLSX to {target}")
PY

echo "CMMD download complete"
find "$METADATA_ROOT" -maxdepth 1 -type f | sort
