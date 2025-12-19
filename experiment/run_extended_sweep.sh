#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Extended Parameter Sweep for Maximum Configuration Testing
# Tests larger input/output lengths, higher concurrency
# Includes both intra-node and inter-node (1PxD) configurations
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"

source "${SCRIPTS_DIR}/00_extended_config.sh"

# ===== EXTENDED SWEEP PARAMETERS =====
# Pushing toward maximum possible values

# Number of prompts (enough for statistical significance)
NUM_PROMPTS_LIST=(200 500)

# Input lengths (extended range for GH200 prefill capacity)
# GH200 can handle large prefills efficiently
INPUT_LEN_LIST=(512 1024 2048 4096)

# Output lengths (test decode throughput)
OUTPUT_LEN_LIST=(128 256 512)

# Concurrency levels (scale with number of decoders)
CONCURRENCY_LIST=(64 128 256)

# Which modes to run
# agg: Aggregated on single node
# pd_intra: Intra-node PD (same node)
# pd_inter_1pXd: Inter-node with X decoders
MODES=("pd_inter_1p1d" "pd_inter_1p2d" "pd_inter_1p4d" "pd_inter_1p8d")

# ===== ENVIRONMENT =====
A100_HOST="${A100_HOST:-172.16.40.99}"
GH200_HOST="${PREFILL_HOST:-172.16.40.79}"

# ===== HELPER FUNCTIONS =====

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_server() {
    local url="$1"
    local max_wait="${2:-120}"
    local waited=0
    
    while ! curl -s --max-time 5 "${url}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            return 1
        fi
    done
    return 0
}

stop_router() {
    pkill -f "sglang_router" 2>/dev/null || true
    sleep 3
}

# Start router for 1PxD configuration
start_1pxd_router() {
    local num_decoders=$1
    local prefill_url="http://${GH200_HOST}:${PREFILL_PORT}"
    
    log "Starting router for 1P${num_decoders}D..."
    
    stop_router
    
    # Build decode endpoints
    local decode_args=""
    for ((i=0; i<num_decoders; i++)); do
        local port=$((DECODE_BASE_PORT + i))
        decode_args="${decode_args} --decode http://127.0.0.1:${port}"
    done
    
    source "${VENV_DIR}/bin/activate"
    nohup python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "${prefill_url}" \
        ${decode_args} \
        --host 0.0.0.0 --port "${ROUTER_PORT}" \
        > /tmp/router_extended.log 2>&1 &
    
    sleep 10
    
    if ! curl -s --max-time 10 "http://127.0.0.1:${ROUTER_PORT}/health" > /dev/null 2>&1; then
        log "ERROR: Router failed to start"
        return 1
    fi
    
    log "Router ready"
}

run_benchmark() {
    local mode="$1"
    local tag="$2"
    local base_url="$3"
    local pd_flag="$4"
    local output_file="${RESULTS_DIR}/${tag}.jsonl"
    
    log "Running: ${tag}"
    
    source "${VENV_DIR}/bin/activate"
    
    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "${base_url}" \
        --dataset-name random \
        --num-prompts "${BENCH_NUM_PROMPTS}" \
        --random-input "${BENCH_INPUT_LEN}" \
        --random-output "${BENCH_OUTPUT_LEN}" \
        --request-rate inf \
        --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
        --output-file "${output_file}" \
        --tag "${tag}" \
        ${pd_flag} || {
            log "WARNING: Benchmark failed for ${tag}"
            return 1
        }
    
    log "Saved: ${output_file}"
}

# ===== MAIN SWEEP =====

main() {
    log "=============================================="
    log "Extended Parameter Sweep"
    log "=============================================="
    log "Modes: ${MODES[*]}"
    log "Num Prompts: ${NUM_PROMPTS_LIST[*]}"
    log "Input Lengths: ${INPUT_LEN_LIST[*]}"
    log "Output Lengths: ${OUTPUT_LEN_LIST[*]}"
    log "Concurrency: ${CONCURRENCY_LIST[*]}"
    log "=============================================="
    
    mkdir -p "${RESULTS_DIR}"
    
    # Check prerequisites
    log "Checking prerequisites..."
    
    # Check prefill server on GH200
    if ! curl -s --max-time 10 "http://${GH200_HOST}:${PREFILL_PORT}/health" > /dev/null 2>&1; then
        log "ERROR: Prefill server not running on GH200 (${GH200_HOST}:${PREFILL_PORT})"
        exit 1
    fi
    log "Prefill server OK"
    
    # Check decode servers
    local max_decoders=8
    local available_decoders=0
    for ((i=0; i<max_decoders; i++)); do
        local port=$((DECODE_BASE_PORT + i))
        if curl -s --max-time 5 "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            available_decoders=$((available_decoders + 1))
        fi
    done
    log "Available decode servers: ${available_decoders}"
    
    if [ $available_decoders -eq 0 ]; then
        log "ERROR: No decode servers running"
        log "Start them with: NUM_DECODERS=8 bash scripts/50_run_multi_decode_a100.sh"
        exit 1
    fi
    
    # Run sweep
    for mode in "${MODES[@]}"; do
        log ""
        log "=============================================="
        log "MODE: ${mode}"
        log "=============================================="
        
        # Extract number of decoders from mode
        local num_decoders=1
        if [[ "$mode" =~ pd_inter_1p([0-9]+)d ]]; then
            num_decoders="${BASH_REMATCH[1]}"
        fi
        
        # Check if we have enough decoders
        if [ "$num_decoders" -gt "$available_decoders" ]; then
            log "WARNING: Need ${num_decoders} decoders but only ${available_decoders} available, skipping..."
            continue
        fi
        
        # Start appropriate router
        if ! start_1pxd_router "$num_decoders"; then
            log "Failed to start router for ${mode}, skipping..."
            continue
        fi
        
        BASE_URL="http://127.0.0.1:${ROUTER_PORT}"
        PD_FLAG="--pd-separated"
        
        # Run parameter sweep for this mode
        for num_prompts in "${NUM_PROMPTS_LIST[@]}"; do
            for input_len in "${INPUT_LEN_LIST[@]}"; do
                for output_len in "${OUTPUT_LEN_LIST[@]}"; do
                    for concurrency in "${CONCURRENCY_LIST[@]}"; do
                        
                        # Set benchmark parameters
                        export BENCH_NUM_PROMPTS="${num_prompts}"
                        export BENCH_INPUT_LEN="${input_len}"
                        export BENCH_OUTPUT_LEN="${output_len}"
                        export BENCH_MAX_CONCURRENCY="${concurrency}"
                        
                        # Create tag
                        TAG="${mode}_n${num_prompts}_in${input_len}_out${output_len}_c${concurrency}"
                        
                        # Skip if output file already exists
                        if [ -f "${RESULTS_DIR}/${TAG}.jsonl" ]; then
                            log "Skipping ${TAG} (already exists)"
                            continue
                        fi
                        
                        log ""
                        log "--- ${TAG} ---"
                        
                        run_benchmark "${mode}" "${TAG}" "${BASE_URL}" "${PD_FLAG}" || true
                        
                        sleep 3
                    done
                done
            done
        done
    done
    
    # Cleanup
    stop_router
    
    log ""
    log "=============================================="
    log "Extended Sweep Complete!"
    log "Results in: ${RESULTS_DIR}"
    log "=============================================="
    
    # Count results
    local result_count=$(ls -1 "${RESULTS_DIR}"/pd_inter_1p*d_*.jsonl 2>/dev/null | wc -l)
    log "Total result files: ${result_count}"
}

# ===== USAGE =====

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --modes M1,M2,...           Modes to test"
    echo "                              (pd_inter_1p1d,pd_inter_1p2d,pd_inter_1p4d,pd_inter_1p8d)"
    echo "  --num-prompts N1,N2,...     Number of prompts (default: 200,500)"
    echo "  --input-lens L1,L2,...      Input lengths (default: 512,1024,2048,4096)"
    echo "  --output-lens L1,L2,...     Output lengths (default: 128,256,512)"
    echo "  --concurrency C1,C2,...     Concurrency levels (default: 64,128,256)"
    echo ""
    echo "Prerequisites:"
    echo "  1. Prefill server on GH200: bash scripts/51_run_prefill_gh200_1pxd.sh"
    echo "  2. Decode servers on A100: NUM_DECODERS=8 bash scripts/50_run_multi_decode_a100.sh"
    echo ""
    echo "Example (quick test):"
    echo "  $0 --modes pd_inter_1p1d,pd_inter_1p8d --num-prompts 100 --input-lens 1024"
    echo ""
    echo "Example (full sweep):"
    echo "  $0"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --modes)
            IFS=',' read -ra MODES <<< "$2"
            shift 2
            ;;
        --num-prompts)
            IFS=',' read -ra NUM_PROMPTS_LIST <<< "$2"
            shift 2
            ;;
        --input-lens)
            IFS=',' read -ra INPUT_LEN_LIST <<< "$2"
            shift 2
            ;;
        --output-lens)
            IFS=',' read -ra OUTPUT_LEN_LIST <<< "$2"
            shift 2
            ;;
        --concurrency)
            IFS=',' read -ra CONCURRENCY_LIST <<< "$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

main


