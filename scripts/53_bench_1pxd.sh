#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Benchmark 1PxD Inter-Node PD Disaggregation
# Run from the A100 node where router is running
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_extended_config.sh"

VENV_DIR="${VENV_DIR:-$HOME/venv_sglang}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/../benchmarks/results}"
mkdir -p "${RESULTS_DIR}"

# Number of decoders (for tagging)
NUM_DECODERS="${NUM_DECODERS:-1}"

# Tag format: pd_1pXd_nN_inI_outO_cC
TAG="${TAG:-pd_1p${NUM_DECODERS}d_n${BENCH_NUM_PROMPTS}_in${BENCH_INPUT_LEN}_out${BENCH_OUTPUT_LEN}_c${BENCH_MAX_CONCURRENCY}}"

ROUTER_URL="http://127.0.0.1:${ROUTER_PORT}"
OUTPUT_FILE="${RESULTS_DIR}/${TAG}.jsonl"

echo "=============================================="
echo "Benchmarking 1P${NUM_DECODERS}D Inter-Node PD"
echo "=============================================="
echo "Router URL: ${ROUTER_URL}"
echo "Num Decoders: ${NUM_DECODERS}"
echo "Prompts: ${BENCH_NUM_PROMPTS}"
echo "Input Length: ${BENCH_INPUT_LEN}"
echo "Output Length: ${BENCH_OUTPUT_LEN}"
echo "Max Concurrency: ${BENCH_MAX_CONCURRENCY}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Check if router is running
if ! curl -s --max-time 5 "${ROUTER_URL}/get_model_info" > /dev/null 2>&1; then
    echo "ERROR: Router not responding at ${ROUTER_URL}"
    echo "Please start the router first with: NUM_DECODERS=${NUM_DECODERS} bash scripts/52_run_router_1pxd.sh"
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
    --dataset-name random \
    --num-prompts "${BENCH_NUM_PROMPTS}" \
    --random-input "${BENCH_INPUT_LEN}" \
    --random-output "${BENCH_OUTPUT_LEN}" \
    --request-rate inf \
    --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
    --pd-separated \
    --output-file "${OUTPUT_FILE}" \
    --tag "${TAG}"

echo ""
echo "=============================================="
echo "1P${NUM_DECODERS}D benchmark finished"
echo "Results: ${OUTPUT_FILE}"
echo "=============================================="


