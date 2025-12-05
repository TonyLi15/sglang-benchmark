#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

# ============================================================
# Start MiniLB Router for PD Disaggregation
# ============================================================

PREFILL_URL="${PREFILL_URL:-http://127.0.0.1:${PREFILL_PORT}}"
DECODE_URL="${DECODE_URL:-http://127.0.0.1:${DECODE_PORT}}"

echo "Starting MiniLB Router for PD Disaggregation"
echo "  Prefill: ${PREFILL_URL}"
echo "  Decode:  ${DECODE_URL}"
echo "  Router:  http://0.0.0.0:${ROUTER_PORT}"

# Kill any existing router
pkill -f sglang_router 2>/dev/null || true
sleep 2

# Activate venv and start router
source "${VENV_DIR}/bin/activate"

python3 -m sglang_router.launch_router \
  --mini-lb \
  --pd-disaggregation \
  --prefill "${PREFILL_URL}" \
  --decode "${DECODE_URL}" \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT}"

