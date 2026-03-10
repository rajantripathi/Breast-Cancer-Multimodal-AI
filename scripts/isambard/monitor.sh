#!/usr/bin/env bash
set -euo pipefail

echo "Jobs:"
squeue -u "$USER" || true
echo
echo "Recent log files:"
find "${PWD}/logs" -maxdepth 1 -type f | sort | tail -n 10 || true

