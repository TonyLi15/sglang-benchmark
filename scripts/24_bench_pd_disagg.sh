#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

# Run from cg1n2, with router running on localhost:${ROUTER_PORT}
source "${VENV_DIR}/bin/activate"

TAG="${TAG:-pd_cg1n1_cg1n2}"
OUT_FILE="${RESULTS_DIR}/${TAG}.jsonl"

BASE_URL="http://127.0.0.1:${ROUTER_PORT}"

echo "Benchmarking PD-DISAGGREGATED server via router at ${BASE_URL}"
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
  --pd-separated \
  --base-url "${BASE_URL}" \
  --output-file "${OUT_FILE}" \
  --tag "${TAG}" \
  --ignore-eos

echo "PD-disaggregated benchmark finished -> ${OUT_FILE}"
