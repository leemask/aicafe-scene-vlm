#!/usr/bin/env bash
# CPU-only development setup; does not download model weights or call paid APIs.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Use Python 3.11 or 3.12"'
command -v node >/dev/null || { echo 'Node.js is required for UI checks.' >&2; exit 1; }
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/check_cloud.sh
