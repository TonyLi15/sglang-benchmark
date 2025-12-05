#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Full Parameter Sweep for All 3 Benchmark Modes
# - Aggregated (single server)
# - Intra-Node PD (prefill + decode on same node)
# - Inter-Node PD (GH200 prefill + A100 decode via NIXL)
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"

# Source common config
source "${SCRIPTS_DIR}/00_common.sh"

# ===== SWEEP PARAMETERS =====
# Adjust these arrays to control the sweep
NUM_PROMPTS_LIST=(50 100 200)
INPUT_LEN_LIST=(128 512 1024)
OUTPUT_LEN_LIST=(64 128 256)
CONCURRENCY_LIST=(8 32 128)

# Which modes to run (comment out to skip)
MODES=("agg" "pd_intra" "pd_inter")

# Inter-node settings
A100_HOST="${A100_HOST:-172.16.40.99}"
GH200_IP="${GH200_IP:-172.16.40.79}"

# ===== HELPER FUNCTIONS =====

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_server() {
    local url="$1"
    local max_wait="${2:-120}"
    local waited=0
    
    log "Waiting for server at ${url}..."
    while ! curl -s --max-time 5 "${url}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            log "ERROR: Server at ${url} not ready after ${max_wait}s"
            return 1
        fi
    done
    log "Server ready at ${url}"
}

stop_all_servers() {
    log "Stopping all servers..."
    docker stop sglang-agg sglang-prefill sglang-decode 2>/dev/null || true
    docker rm sglang-agg sglang-prefill sglang-decode 2>/dev/null || true
    ssh "${A100_HOST}" "docker stop sglang-decode 2>/dev/null; docker rm sglang-decode 2>/dev/null; pkill -f sglang_router" 2>/dev/null || true
    pkill -f sglang_router 2>/dev/null || true
    sleep 5
}

# ===== SERVER MANAGEMENT =====

start_agg_server() {
    log "Starting aggregated server..."
    stop_all_servers
    bash "${SCRIPTS_DIR}/10_run_agg_server.sh"
    wait_for_server "http://127.0.0.1:30000"
}

start_intra_node_pd() {
    log "Starting intra-node PD servers..."
    stop_all_servers
    bash "${SCRIPTS_DIR}/30_run_intra_node_pd.sh"
    wait_for_server "http://127.0.0.1:${PREFILL_PORT}"
    wait_for_server "http://127.0.0.1:${DECODE_PORT}"
    
    # Start router
    log "Starting router..."
    source "${VENV_DIR}/bin/activate"
    nohup python3 -m sglang_router.launch_router \
        --mini-lb --pd-disaggregation \
        --prefill "http://127.0.0.1:${PREFILL_PORT}" \
        --decode "http://127.0.0.1:${DECODE_PORT}" \
        --host 0.0.0.0 --port "${ROUTER_PORT}" \
        > /tmp/router_intra.log 2>&1 &
    sleep 10
    wait_for_server "http://127.0.0.1:${ROUTER_PORT}"
}

start_inter_node_pd() {
    log "Starting inter-node PD servers (GH200 + A100)..."
    stop_all_servers
    
    # Start prefill on GH200 with NIXL
    log "Starting prefill on GH200..."
    docker run -d \
        --name sglang-prefill \
        --gpus all \
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
            --port 30000 \
            --mem-fraction-static 0.9 \
            --disaggregation-mode prefill \
            --disaggregation-transfer-backend nixl
    
    # Start decode on A100 with NIXL
    log "Starting decode on A100..."
    ssh "${A100_HOST}" "docker run -d \
        --name sglang-decode \
        --gpus 'device=0' \
        --ipc=host \
        --shm-size=32g \
        --privileged \
        --network=host \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -e HF_HOME=/root/.cache/huggingface \
        ${SGLANG_IMAGE_X86} \
        python3 -m sglang.launch_server \
            --model-path ${MODEL_PATH} \
            --host 0.0.0.0 \
            --port 30000 \
            --mem-fraction-static 0.9 \
            --disaggregation-mode decode \
            --disaggregation-transfer-backend nixl"
    
    # Wait for servers
    wait_for_server "http://127.0.0.1:30000" 120
    sleep 30  # Extra time for A100
    
    # Start router on A100
    log "Starting router on A100..."
    ssh -f "${A100_HOST}" "source ~/venv_sglang/bin/activate && \
        nohup python3 -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill http://${GH200_IP}:30000 \
            --decode http://127.0.0.1:30000 \
            --host 0.0.0.0 --port 8000 \
            > /tmp/router_inter.log 2>&1 &"
    sleep 10
}

# ===== BENCHMARK FUNCTIONS =====

run_benchmark() {
    local mode="$1"
    local tag="$2"
    local base_url="$3"
    local pd_flag="$4"
    
    log "Running benchmark: ${tag}"
    
    source "${VENV_DIR}/bin/activate"
    
    local output_file="${RESULTS_DIR}/${tag}.jsonl"
    
    python3 -m sglang.bench_serving \
        --backend sglang \
        --dataset-name random \
        --num-prompts "${BENCH_NUM_PROMPTS}" \
        --random-input "${BENCH_INPUT_LEN}" \
        --random-output "${BENCH_OUTPUT_LEN}" \
        --request-rate inf \
        --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
        --base-url "${base_url}" \
        --output-file "${output_file}" \
        --tag "${tag}" \
        ${pd_flag}
    
    log "Results saved to: ${output_file}"
}

# ===== MAIN SWEEP LOGIC =====

main() {
    log "=========================================="
    log "Starting Full Parameter Sweep"
    log "=========================================="
    log "Modes: ${MODES[*]}"
    log "Num Prompts: ${NUM_PROMPTS_LIST[*]}"
    log "Input Lengths: ${INPUT_LEN_LIST[*]}"
    log "Output Lengths: ${OUTPUT_LEN_LIST[*]}"
    log "Concurrency: ${CONCURRENCY_LIST[*]}"
    log "=========================================="
    
    mkdir -p "${RESULTS_DIR}"
    
    for mode in "${MODES[@]}"; do
        log ""
        log "=========================================="
        log "MODE: ${mode}"
        log "=========================================="
        
        # Start appropriate servers
        case "${mode}" in
            "agg")
                start_agg_server
                BASE_URL="http://127.0.0.1:30000"
                PD_FLAG=""
                ;;
            "pd_intra")
                start_intra_node_pd
                BASE_URL="http://127.0.0.1:${ROUTER_PORT}"
                PD_FLAG="--pd-separated"
                ;;
            "pd_inter")
                start_inter_node_pd
                BASE_URL="http://${A100_HOST}:8000"
                PD_FLAG="--pd-separated"
                ;;
            *)
                log "Unknown mode: ${mode}"
                continue
                ;;
        esac
        
        # Run sweep for this mode
        for num_prompts in "${NUM_PROMPTS_LIST[@]}"; do
            for input_len in "${INPUT_LEN_LIST[@]}"; do
                for output_len in "${OUTPUT_LEN_LIST[@]}"; do
                    for concurrency in "${CONCURRENCY_LIST[@]}"; do
                        
                        # Export benchmark parameters
                        export BENCH_NUM_PROMPTS="${num_prompts}"
                        export BENCH_INPUT_LEN="${input_len}"
                        export BENCH_OUTPUT_LEN="${output_len}"
                        export BENCH_MAX_CONCURRENCY="${concurrency}"
                        
                        # Create unique tag
                        TAG="${mode}_n${num_prompts}_in${input_len}_out${output_len}_c${concurrency}"
                        
                        log ""
                        log "--- ${TAG} ---"
                        
                        # Run benchmark
                        run_benchmark "${mode}" "${TAG}" "${BASE_URL}" "${PD_FLAG}" || {
                            log "WARNING: Benchmark failed for ${TAG}, continuing..."
                        }
                        
                        # Small delay between runs
                        sleep 5
                        
                    done
                done
            done
        done
        
        log "Completed mode: ${mode}"
    done
    
    # Cleanup
    stop_all_servers
    
    log ""
    log "=========================================="
    log "Full sweep completed!"
    log "Results in: ${RESULTS_DIR}"
    log "=========================================="
    
    # Generate plots
    log "Generating plots..."
    python3 "${REPO_ROOT}/benchmarks/plot_benchmarks.py" || true
}

# ===== ENTRY POINT =====

# Parse command line arguments
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --modes MODE1,MODE2,...     Modes to run (agg,pd_intra,pd_inter)"
            echo "  --num-prompts N1,N2,...     Number of prompts"
            echo "  --input-lens L1,L2,...      Input lengths"
            echo "  --output-lens L1,L2,...     Output lengths"
            echo "  --concurrency C1,C2,...     Concurrency levels"
            echo ""
            echo "Example:"
            echo "  $0 --modes agg,pd_intra --num-prompts 50,100 --input-lens 128,512"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

main

