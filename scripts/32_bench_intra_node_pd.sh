#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

# ============================================================
# Benchmark Intra-Node PD Disaggregation
# ============================================================

TAG="${TAG:-pd_intra_node}"
ROUTER_URL="http://127.0.0.1:${ROUTER_PORT}"
OUTPUT_FILE="${RESULTS_DIR}/${TAG}.jsonl"

echo "=============================================="
echo "Benchmarking Intra-Node PD Disaggregation"
echo "=============================================="
echo "Router URL: ${ROUTER_URL}"
echo "Prompts: ${BENCH_NUM_PROMPTS}"
echo "Input Length: ${BENCH_INPUT_LEN}"
echo "Output Length: ${BENCH_OUTPUT_LEN}"
echo "Max Concurrency: ${BENCH_MAX_CONCURRENCY}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Check if router is running
if ! curl -s --max-time 5 "${ROUTER_URL}/get_model_info" > /dev/null 2>&1; then
    echo "ERROR: Router not responding at ${ROUTER_URL}"
    echo "Please start the router first: bash scripts/31_start_router.sh"
    exit 1
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# Run benchmark
python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url "${ROUTER_URL}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" \
    --random-input "${BENCH_INPUT_LEN}" \
    --random-output "${BENCH_OUTPUT_LEN}" \
    --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --pd-separated \
    --output-file "${OUTPUT_FILE}"

echo ""
echo "=============================================="
echo "Intra-Node PD benchmark finished -> ${OUTPUT_FILE}"
echo "=============================================="

