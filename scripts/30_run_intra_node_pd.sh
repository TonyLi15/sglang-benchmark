#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

# ============================================================
# Intra-Node PD Disaggregation Setup
# Runs both prefill and decode servers on the same node
# ============================================================

echo "=============================================="
echo "Intra-Node PD Disaggregation Setup"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Image: ${SGLANG_IMAGE}"
echo "IB Device: ${IB_DEVICE}"
echo "Prefill Port: ${PREFILL_PORT}"
echo "Decode Port: ${DECODE_PORT}"
echo "Router Port: ${ROUTER_PORT}"
echo "=============================================="

# Stop any existing containers
echo "[1/4] Stopping existing containers..."
docker stop sglang-prefill sglang-decode sglang-agg 2>/dev/null || true
docker rm sglang-prefill sglang-decode sglang-agg 2>/dev/null || true

# Start prefill server
echo "[2/4] Starting prefill server on port ${PREFILL_PORT}..."
docker run -d \
  --name sglang-prefill \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --privileged \
  --network=host \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  "${SGLANG_IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PREFILL_PORT}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --disaggregation-mode prefill \
    --disaggregation-ib-device "${IB_DEVICE}" \
    --disaggregation-bootstrap-port 8998

# Start decode server
echo "[3/4] Starting decode server on port ${DECODE_PORT}..."
docker run -d \
  --name sglang-decode \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --privileged \
  --network=host \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  "${SGLANG_IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${DECODE_PORT}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --disaggregation-mode decode \
    --disaggregation-ib-device "${IB_DEVICE}"

# Wait for servers to be ready
echo "[4/4] Waiting for servers to initialize (60 seconds)..."
sleep 60

# Check server status
echo ""
echo "Checking server status..."
echo "--- Prefill Server ---"
docker logs sglang-prefill 2>&1 | grep -E "fired up|ERROR|error" | tail -3 || echo "(check logs manually)"
echo ""
echo "--- Decode Server ---"
docker logs sglang-decode 2>&1 | grep -E "fired up|ERROR|error" | tail -3 || echo "(check logs manually)"

echo ""
echo "=============================================="
echo "Intra-Node PD servers started!"
echo ""
echo "To start the router, run:"
echo "  bash scripts/31_start_router.sh"
echo ""
echo "Or manually:"
echo "  source ~/venv_sglang/bin/activate"
echo "  python3 -m sglang_router.launch_router \\"
echo "    --mini-lb --pd-disaggregation \\"
echo "    --prefill http://127.0.0.1:${PREFILL_PORT} \\"
echo "    --decode http://127.0.0.1:${DECODE_PORT} \\"
echo "    --host 0.0.0.0 --port ${ROUTER_PORT}"
echo "=============================================="

