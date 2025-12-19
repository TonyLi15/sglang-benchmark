#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Multi-Decode Server Setup for A100 Cluster (1PxD)
# Run this on the A100 node (172.16.40.99)
# Starts x decode servers on x GPUs (x = 1 to 8)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_extended_config.sh"

# Number of decode servers to start (1-8)
NUM_DECODERS="${NUM_DECODERS:-1}"

# Validate NUM_DECODERS
if [ "$NUM_DECODERS" -lt 1 ] || [ "$NUM_DECODERS" -gt 8 ]; then
    echo "ERROR: NUM_DECODERS must be between 1 and 8"
    exit 1
fi

# Force x86 image for A100 node (override any ARM64 setting)
SGLANG_IMAGE="${SGLANG_IMAGE_X86:-lmsysorg/sglang:latest}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
# For inter-node PD, always use NIXL (cross-fabric RDMA support)
TRANSFER_BACKEND="${INTER_NODE_TRANSFER_BACKEND:-nixl}"

echo "=============================================="
echo "Starting ${NUM_DECODERS} Decode Server(s) on A100"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Image: ${SGLANG_IMAGE}"
echo "Transfer Backend: ${TRANSFER_BACKEND}"
echo "=============================================="

# Stop any existing decode containers
echo "Stopping existing decode containers..."
for i in $(seq 0 7); do
    docker stop "sglang-decode-${i}" 2>/dev/null || true
    docker rm "sglang-decode-${i}" 2>/dev/null || true
done

# Start decode servers
for ((i=0; i<NUM_DECODERS; i++)); do
    PORT=$((DECODE_BASE_PORT + i))
    CONTAINER_NAME="sglang-decode-${i}"
    
    echo ""
    echo "Starting decode server ${i}: GPU=${i}, Port=${PORT}"
    
    # Use CUDA_VISIBLE_DEVICES for proper GPU isolation (no --gpus flag needed with nvidia runtime)
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --runtime=nvidia \
        --ipc=host \
        --shm-size=32g \
        --privileged \
        --network=host \
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        -e CUDA_VISIBLE_DEVICES="${i}" \
        "${SGLANG_IMAGE}" \
        python3 -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --mem-fraction-static 0.9 \
            --disaggregation-mode decode \
            --disaggregation-transfer-backend "${TRANSFER_BACKEND}"
    
    echo "  Container: ${CONTAINER_NAME}"
    echo "  Logs: docker logs -f ${CONTAINER_NAME}"
done

echo ""
echo "=============================================="
echo "Started ${NUM_DECODERS} decode server(s)"
echo "Ports: $(for ((i=0; i<NUM_DECODERS; i++)); do echo -n "$((DECODE_BASE_PORT + i)) "; done)"
echo ""
echo "Wait for servers to be ready (check logs):"
for ((i=0; i<NUM_DECODERS; i++)); do
    echo "  docker logs -f sglang-decode-${i}"
done
echo "=============================================="

