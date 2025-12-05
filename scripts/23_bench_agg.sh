#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

# Run from cg1n2, assuming aggregated server running on cg1n1:30000
source "${VENV_DIR}/bin/activate"

TAG="${TAG:-agg_cg1n1}"
OUT_FILE="${RESULTS_DIR}/${TAG}.jsonl"

BASE_URL="http://${PREFILL_HOST}:${PREFILL_PORT}"

echo "Benchmarking AGGREGATED server at ${BASE_URL}"
echo "    Tag: ${TAG}"
echo "    Output: ${OUT_FILE}"

python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts "${BENCH_NUM_PROMPTS}" \
  --random-input "${BENCH_INPUT_LEN}" \
  --random-output "${BENCH_OUTPUT_LEN}" \
  --request-rate inf \
  --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
  --base-url "${BASE_URL}" \
  --output-file "${OUT_FILE}" \
  --tag "${TAG}"

echo "Aggregated benchmark finished -> ${OUT_FILE}"
