#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1PxD Scaling Sweep: Benchmark with varying number of decoders
# Tests 1P1D, 1P2D, 1P4D, 1P8D configurations
# 
# This script should be run on the A100 node after:
# 1. Prefill server is running on GH200
# 2. All decode servers are started (NUM_DECODERS=8)
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"

source "${SCRIPTS_DIR}/00_extended_config.sh"

# ===== SWEEP CONFIGURATION =====

# Number of decoders to test (scaling study)
DECODER_COUNTS=(1 2 4 8)

# Benchmark parameters (can be overridden via env)
# Using extended parameter ranges for stress testing
SWEEP_NUM_PROMPTS="${SWEEP_NUM_PROMPTS:-200}"
SWEEP_INPUT_LEN="${SWEEP_INPUT_LEN:-1024}"
SWEEP_OUTPUT_LEN="${SWEEP_OUTPUT_LEN:-256}"
SWEEP_CONCURRENCY="${SWEEP_CONCURRENCY:-128}"

# Prefill server location
PREFILL_URL="http://${PREFILL_HOST}:${PREFILL_PORT}"

# ===== HELPER FUNCTIONS =====

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_server() {
    local url="$1"
    local name="$2"
    local max_wait="${3:-60}"
    local waited=0
    
    log "Waiting for ${name} at ${url}..."
    while ! curl -s --max-time 5 "${url}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            log "ERROR: ${name} at ${url} not ready after ${max_wait}s"
            return 1
        fi
    done
    log "${name} ready"
}

stop_router() {
    log "Stopping router..."
    pkill -f "sglang_router" 2>/dev/null || true
    sleep 3
}

start_router() {
    local num_decoders=$1
    
    log "Starting router for 1P${num_decoders}D configuration..."
    
    # Build decode endpoints
    local decode_args=""
    for ((i=0; i<num_decoders; i++)); do
        local port=$((DECODE_BASE_PORT + i))
        decode_args="${decode_args} --decode http://127.0.0.1:${port}"
    done
    
    # Start router in background
    source "${VENV_DIR}/bin/activate"
    nohup python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "${PREFILL_URL}" \
        ${decode_args} \
        --host 0.0.0.0 --port "${ROUTER_PORT}" \
        > /tmp/router_1p${num_decoders}d.log 2>&1 &
    
    sleep 10
    
    # Verify router is running
    if ! curl -s --max-time 10 "http://127.0.0.1:${ROUTER_PORT}/health" > /dev/null 2>&1; then
        log "ERROR: Router failed to start. Check /tmp/router_1p${num_decoders}d.log"
        return 1
    fi
    
    log "Router started for 1P${num_decoders}D"
}

run_benchmark() {
    local num_decoders=$1
    local tag="pd_1p${num_decoders}d_n${SWEEP_NUM_PROMPTS}_in${SWEEP_INPUT_LEN}_out${SWEEP_OUTPUT_LEN}_c${SWEEP_CONCURRENCY}"
    local output_file="${RESULTS_DIR}/${tag}.jsonl"
    
    log "Running benchmark: ${tag}"
    
    source "${VENV_DIR}/bin/activate"
    
    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://127.0.0.1:${ROUTER_PORT}" \
        --dataset-name random \
        --num-prompts "${SWEEP_NUM_PROMPTS}" \
        --random-input "${SWEEP_INPUT_LEN}" \
        --random-output "${SWEEP_OUTPUT_LEN}" \
        --request-rate inf \
        --max-concurrency "${SWEEP_CONCURRENCY}" \
        --pd-separated \
        --output-file "${output_file}" \
        --tag "${tag}"
    
    log "Results saved to: ${output_file}"
}

# ===== MAIN =====

main() {
    log "=============================================="
    log "1PxD Scaling Sweep"
    log "=============================================="
    log "Decoder counts: ${DECODER_COUNTS[*]}"
    log "Num prompts: ${SWEEP_NUM_PROMPTS}"
    log "Input length: ${SWEEP_INPUT_LEN}"
    log "Output length: ${SWEEP_OUTPUT_LEN}"
    log "Concurrency: ${SWEEP_CONCURRENCY}"
    log "=============================================="
    
    mkdir -p "${RESULTS_DIR}"
    
    # Verify prefill server is running
    log "Checking prefill server..."
    if ! curl -s --max-time 10 "${PREFILL_URL}/health" > /dev/null 2>&1; then
        log "ERROR: Prefill server not running at ${PREFILL_URL}"
        log "Start it on GH200 with: bash scripts/51_run_prefill_gh200_1pxd.sh"
        exit 1
    fi
    log "Prefill server OK"
    
    # Verify decode servers are running (check max needed)
    local max_decoders=${DECODER_COUNTS[-1]}
    log "Checking ${max_decoders} decode servers..."
    for ((i=0; i<max_decoders; i++)); do
        local port=$((DECODE_BASE_PORT + i))
        if ! curl -s --max-time 5 "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            log "ERROR: Decode server ${i} not running at port ${port}"
            log "Start decode servers with: NUM_DECODERS=${max_decoders} bash scripts/50_run_multi_decode_a100.sh"
            exit 1
        fi
    done
    log "All ${max_decoders} decode servers OK"
    
    # Run sweep for each decoder count
    for num_decoders in "${DECODER_COUNTS[@]}"; do
        log ""
        log "=============================================="
        log "Testing 1P${num_decoders}D configuration"
        log "=============================================="
        
        # Stop existing router
        stop_router
        
        # Start router with this many decoders
        if ! start_router "${num_decoders}"; then
            log "Failed to start router for 1P${num_decoders}D, skipping..."
            continue
        fi
        
        # Run benchmark
        if ! run_benchmark "${num_decoders}"; then
            log "Benchmark failed for 1P${num_decoders}D, continuing..."
        fi
        
        # Brief pause between tests
        sleep 5
    done
    
    # Cleanup
    stop_router
    
    log ""
    log "=============================================="
    log "1PxD Scaling Sweep Complete!"
    log "Results in: ${RESULTS_DIR}"
    log "=============================================="
    
    # List generated files
    log ""
    log "Generated files:"
    ls -la "${RESULTS_DIR}"/pd_1p*d_*.jsonl 2>/dev/null || echo "No files found"
}

# ===== ENTRY POINT =====

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --decoders)
            IFS=',' read -ra DECODER_COUNTS <<< "$2"
            shift 2
            ;;
        --num-prompts)
            SWEEP_NUM_PROMPTS="$2"
            shift 2
            ;;
        --input-len)
            SWEEP_INPUT_LEN="$2"
            shift 2
            ;;
        --output-len)
            SWEEP_OUTPUT_LEN="$2"
            shift 2
            ;;
        --concurrency)
            SWEEP_CONCURRENCY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --decoders N1,N2,...     Number of decoders to test (default: 1,2,4,8)"
            echo "  --num-prompts N          Number of prompts (default: 200)"
            echo "  --input-len L            Input token length (default: 1024)"
            echo "  --output-len L           Output token length (default: 256)"
            echo "  --concurrency C          Max concurrency (default: 128)"
            echo ""
            echo "Prerequisites:"
            echo "  1. Prefill server running on GH200"
            echo "  2. Decode servers running on A100 (NUM_DECODERS=8)"
            echo ""
            echo "Example:"
            echo "  $0 --decoders 1,2,4,8 --concurrency 256"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

main


