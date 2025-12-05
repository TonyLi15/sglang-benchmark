#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_common.sh"

# ============================================================
# Full Intra-Node PD Disaggregation Benchmark
# Starts servers, router, runs benchmark, and cleans up
# ============================================================

TAG="${TAG:-pd_intra_node}"

echo "=============================================="
echo "Full Intra-Node PD Disaggregation Benchmark"
echo "=============================================="

# Step 1: Start PD servers
echo ""
echo "[Step 1/4] Starting PD servers..."
bash "$(dirname "$0")/30_run_intra_node_pd.sh"

# Step 2: Start router in background
echo ""
echo "[Step 2/4] Starting router in background..."
source "${VENV_DIR}/bin/activate"
pkill -f sglang_router 2>/dev/null || true
sleep 2

nohup python3 -m sglang_router.launch_router \
  --mini-lb \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:${PREFILL_PORT}" \
  --decode "http://127.0.0.1:${DECODE_PORT}" \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT}" > /tmp/router.log 2>&1 &

echo "Router starting... waiting 10 seconds"
sleep 10

# Verify router is running
if ! curl -s --max-time 5 "http://127.0.0.1:${ROUTER_PORT}/get_model_info" > /dev/null 2>&1; then
    echo "ERROR: Router failed to start. Check /tmp/router.log"
    exit 1
fi
echo "Router is ready!"

# Step 3: Run benchmark
echo ""
echo "[Step 3/4] Running benchmark..."
TAG="${TAG}" bash "$(dirname "$0")/32_bench_intra_node_pd.sh"

# Step 4: Cleanup (optional - comment out to keep servers running)
echo ""
echo "[Step 4/4] Cleaning up..."
pkill -f sglang_router 2>/dev/null || true
# docker stop sglang-prefill sglang-decode 2>/dev/null || true

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results: ${RESULTS_DIR}/${TAG}.jsonl"
echo ""
echo "To generate plots, run:"
echo "  python3 benchmarks/plot_benchmarks.py"
echo "=============================================="

