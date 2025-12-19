#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Router for 1PxD Configuration
# Run this on the A100 node after decode servers are ready
# Routes requests to 1 Prefill + x Decode servers
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_extended_config.sh"

# Number of decode servers (must match what was started)
NUM_DECODERS="${NUM_DECODERS:-1}"

# Prefill URL (GH200 node)
PREFILL_URL="http://${PREFILL_HOST}:${PREFILL_PORT}"

# Build decode URL list
DECODE_URLS=""
for ((i=0; i<NUM_DECODERS; i++)); do
    PORT=$((DECODE_BASE_PORT + i))
    if [ -n "$DECODE_URLS" ]; then
        DECODE_URLS="${DECODE_URLS},"
    fi
    DECODE_URLS="${DECODE_URLS}http://127.0.0.1:${PORT}"
done

echo "=============================================="
echo "Starting Router for 1P${NUM_DECODERS}D Configuration"
echo "=============================================="
echo "Prefill: ${PREFILL_URL}"
echo "Decode endpoints: ${DECODE_URLS}"
echo "Router port: ${ROUTER_PORT}"
echo "=============================================="

# Kill existing router
pkill -f "sglang_router" 2>/dev/null || true
sleep 2

# Activate venv
VENV_DIR="${VENV_DIR:-$HOME/venv_sglang}"
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
fi

# Check if prefill server is reachable
echo "Checking prefill server..."
if ! curl -s --max-time 10 "${PREFILL_URL}/health" > /dev/null 2>&1; then
    echo "WARNING: Prefill server at ${PREFILL_URL} not responding"
    echo "Make sure the prefill server is running on GH200"
fi

# Check decode servers
echo "Checking decode servers..."
for ((i=0; i<NUM_DECODERS; i++)); do
    PORT=$((DECODE_BASE_PORT + i))
    if ! curl -s --max-time 5 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "WARNING: Decode server ${i} at port ${PORT} not responding"
    else
        echo "  Decode server ${i} (port ${PORT}): OK"
    fi
done

# Build router command
# Note: sglang_router supports multiple decode endpoints
ROUTER_CMD="python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill ${PREFILL_URL}"

# Add decode endpoints
for ((i=0; i<NUM_DECODERS; i++)); do
    PORT=$((DECODE_BASE_PORT + i))
    ROUTER_CMD="${ROUTER_CMD} --decode http://127.0.0.1:${PORT}"
done

ROUTER_CMD="${ROUTER_CMD} --host 0.0.0.0 --port ${ROUTER_PORT}"

echo ""
echo "Starting router..."
echo "Command: ${ROUTER_CMD}"
echo ""

# Run router in foreground (Ctrl+C to stop)
# For background, use: nohup ... > /tmp/router_1pxd.log 2>&1 &
eval "${ROUTER_CMD}"


