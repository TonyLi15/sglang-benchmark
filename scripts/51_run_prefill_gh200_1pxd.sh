#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Prefill Server for GH200 (1PxD Inter-Node Configuration)
# Run this on cg1n1 (172.16.40.79)
# Configured to work with multiple decode servers on A100
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_extended_config.sh"

SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:dev-arm64}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
# For inter-node PD, always use NIXL (cross-fabric RDMA support)
TRANSFER_BACKEND="${INTER_NODE_TRANSFER_BACKEND:-nixl}"

CONTAINER_NAME="sglang-prefill"

echo "=============================================="
echo "Starting Prefill Server on GH200 (1PxD Mode)"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Image: ${SGLANG_IMAGE}"
echo "Port: ${PREFILL_PORT}"
echo "Transfer Backend: ${TRANSFER_BACKEND}"
echo "=============================================="

# Stop existing containers
echo "Stopping existing containers..."
docker stop sglang-prefill sglang-decode sglang-agg 2>/dev/null || true
docker rm sglang-prefill sglang-decode sglang-agg 2>/dev/null || true

# Start prefill server with NIXL backend
echo "Starting prefill server..."
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
echo "=============================================="
echo "Prefill server starting on port ${PREFILL_PORT}"
echo "Check logs: docker logs -f ${CONTAINER_NAME}"
echo ""
echo "Next steps:"
echo "1. Wait for this server to be ready"
echo "2. On A100 node, run: NUM_DECODERS=x bash scripts/50_run_multi_decode_a100.sh"
echo "3. On A100 node, run: NUM_DECODERS=x bash scripts/52_run_router_1pxd.sh"
echo "=============================================="

