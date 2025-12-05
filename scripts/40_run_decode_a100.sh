#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Decode Server for A100 (Inter-Node PD Disaggregation)
# Run this on the A100 node (172.16.40.99)
# Uses NIXL backend for cross-fabric RDMA (IB/RoCE)
# ============================================================

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
DECODE_PORT="${DECODE_PORT:-30000}"
GPU_ID="${GPU_ID:-0}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-nixl}"  # NIXL for cross-fabric support

CONTAINER_NAME="sglang-decode"

echo "=============================================="
echo "Starting Decode Server on A100"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Image: ${SGLANG_IMAGE}"
echo "Port: ${DECODE_PORT}"
echo "Transfer Backend: ${TRANSFER_BACKEND}"
echo "GPU: ${GPU_ID}"
echo "=============================================="

# Stop existing container
docker stop "${CONTAINER_NAME}" 2>/dev/null || true
docker rm "${CONTAINER_NAME}" 2>/dev/null || true

# Start decode server with NIXL backend
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus "device=${GPU_ID}" \
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
    --mem-fraction-static 0.9 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend "${TRANSFER_BACKEND}"

echo ""
echo "Decode server starting on port ${DECODE_PORT}"
echo "Check logs: docker logs -f ${CONTAINER_NAME}"

