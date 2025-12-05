#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Prefill Server for GH200 (Inter-Node PD Disaggregation)
# Run this on cg1n1 (172.16.40.79)
# Uses NIXL backend for cross-fabric RDMA (IB/RoCE)
# ============================================================

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:dev-arm64}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-nixl}"  # NIXL for cross-fabric support

CONTAINER_NAME="sglang-prefill"

echo "=============================================="
echo "Starting Prefill Server on GH200"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Image: ${SGLANG_IMAGE}"
echo "Port: ${PREFILL_PORT}"
echo "Transfer Backend: ${TRANSFER_BACKEND}"
echo "=============================================="

# Stop existing containers
docker stop sglang-prefill sglang-decode sglang-agg 2>/dev/null || true
docker rm sglang-prefill sglang-decode sglang-agg 2>/dev/null || true

# Start prefill server with NIXL backend
docker run -d \
  --name "${CONTAINER_NAME}" \
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
    --mem-fraction-static 0.9 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"

echo ""
echo "Prefill server starting on port ${PREFILL_PORT}"
echo "Check logs: docker logs -f ${CONTAINER_NAME}"

