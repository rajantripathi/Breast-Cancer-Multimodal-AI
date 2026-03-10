#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-u6ef.aip2.isambard}"
REMOTE_ROOT="${REMOTE_ROOT:-\$SCRATCH/breast-cancer-multimodal-ai/repo}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rsync -avz --exclude='.venv' --exclude='.git' "$LOCAL_ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/"

