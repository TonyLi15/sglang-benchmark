#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

CONTAINER_NAME="${CONTAINER_NAME:-sglang-agg}"

echo "Starting aggregated SGLang server container: ${CONTAINER_NAME}"
echo "    Model: ${MODEL_PATH}"
echo "    Image: ${SGLANG_IMAGE}"
echo "    Exposed port: ${PREFILL_PORT} -> container:30000"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run -d   --name "${CONTAINER_NAME}"   --gpus all   --ipc=host   --shm-size=32g   -p "${PREFILL_PORT}:30000"   -v "${HF_CACHE_DIR}:/root/.cache/huggingface"   -e HF_HOME=/root/.cache/huggingface   "${SGLANG_IMAGE}"   python3 -m sglang.launch_server     --model-path "${MODEL_PATH}"     --host 0.0.0.0     --port 30000     --mem-fraction-static 0.9

echo "Aggregated server started on port ${PREFILL_PORT} (container=${CONTAINER_NAME})"
echo "   Test on this node: curl http://localhost:${PREFILL_PORT}/get_model_info"
