#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

# This script should be run on cg1n2 (decode node) and will stay in the foreground.

source "${VENV_DIR}/bin/activate"

echo "Starting PD-disaggregation router (sglang-router)..."
echo "    Prefill: http://${PREFILL_HOST}:${PREFILL_PORT}"
echo "    Decode : http://127.0.0.1:${DECODE_PORT}"
echo "    Listen : 0.0.0.0:${ROUTER_PORT}"

exec python3 -m sglang_router.launch_router     --mini-lb     --pd-disaggregation     --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}"     --decode "http://127.0.0.1:${DECODE_PORT}"     --host 0.0.0.0     --port "${ROUTER_PORT}"
