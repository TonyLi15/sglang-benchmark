#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

echo "Creating Python venv at ${VENV_DIR} on this host (router + benchmarks)..."

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip

# Core tools: sglang client, router, triton, plotting libs
pip install   "sglang>=0.5.5"   "sglang-router"   "triton"   "matplotlib"   "pandas"

echo "Host venv ready at ${VENV_DIR} (sglang, sglang-router, triton, matplotlib, pandas installed)"
