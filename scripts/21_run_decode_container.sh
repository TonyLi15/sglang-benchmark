#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

CONTAINER_NAME="${CONTAINER_NAME:-sglang-decode}"

echo "Starting DECODE-only SGLang server container: ${CONTAINER_NAME}"
echo "    Model: ${MODEL_PATH}"
echo "    Image: ${SGLANG_IMAGE}"
echo "    Exposed port: 30001 -> container:30000"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0"' \
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
    --port 30001 \
    --mem-fraction-static 0.8 \
    --disaggregation-mode decode \
    --disaggregation-ib-device mlx5_0

echo "Decode server started on port 30001"
