#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Benchmark Inter-Node PD Disaggregation
# Run from the decode node (A100) where router is running
# ============================================================

VENV_DIR="${VENV_DIR:-$HOME/venv_sglang}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
RESULTS_DIR="${RESULTS_DIR:-$(dirname "$0")/../benchmarks/results}"
mkdir -p "${RESULTS_DIR}"

TAG="${TAG:-pd_inter_node}"
ROUTER_URL="http://127.0.0.1:${ROUTER_PORT}"
OUTPUT_FILE="${RESULTS_DIR}/${TAG}.jsonl"

BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-512}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-200}"

echo "=============================================="
echo "Benchmarking Inter-Node PD Disaggregation"
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
    echo "Please start the router first"
    exit 1
fi

# Activate venv if exists
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
fi

# Run benchmark
python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url "${ROUTER_URL}" \
    --num-prompts "${BENCH_NUM_PROMPTS}" \
    --random-input "${BENCH_INPUT_LEN}" \
    --random-output "${BENCH_OUTPUT_LEN}" \
    --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --pd-separated \
    --output-file "${OUTPUT_FILE}" \
    --tag "${TAG}"

echo ""
echo "=============================================="
echo "Inter-Node PD benchmark finished -> ${OUTPUT_FILE}"
echo "=============================================="

