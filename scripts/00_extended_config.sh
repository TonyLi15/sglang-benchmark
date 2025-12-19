#!/usr/bin/env bash
# ============================================================
# Extended Configuration for Maximum Parameter Benchmarks
# Includes settings for 1PxD (1 Prefill, x Decoders) experiments
# ============================================================

set -euo pipefail

# Source base config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

# ===== EXTENDED BENCHMARK PARAMETERS =====
# These are designed to push the system to maximum capacity

# Model Configuration
# Qwen2.5-3B: ~6GB weights, 32K context, good for testing
# For larger models, adjust accordingly
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_CONTEXT_LENGTH=32768  # Qwen2.5 native context

# ===== GPU MEMORY ESTIMATES =====
# GH200: 96GB HBM3
# A100: 80GB HBM2e per GPU
#
# Memory breakdown (approximate):
#   - Model weights (3B @ fp16): ~6GB
#   - KV Cache per token: ~0.05MB (varies by model)
#   - CUDA graphs: ~1GB
#   - Overhead: ~2-3GB
#
# For Qwen2.5-3B on A100 80GB with 0.9 mem_fraction:
#   Available for KV: ~80 * 0.9 - 6 - 1 - 2 = ~63GB
#   Max tokens: ~63GB / 0.05MB ≈ 1.26M tokens
#   Per-request with 4K context: ~315 concurrent requests

# ===== MAXIMUM BENCHMARK PARAMETERS =====
# Extended parameter ranges for comprehensive testing

# Number of prompts (statistical significance)
MAX_NUM_PROMPTS=(100 200 500)

# Input token lengths (push toward context limit)
# GH200 can handle larger prefills efficiently
MAX_INPUT_LEN=(512 1024 2048 4096)

# Output token lengths
MAX_OUTPUT_LEN=(128 256 512 1024)

# Concurrency levels (GPU utilization)
# Higher concurrency = better batching = higher throughput
MAX_CONCURRENCY=(32 64 128 256)

# ===== 1PxD CONFIGURATION =====
# Multi-decoder setup: 1 Prefill (GH200) + x Decoders (A100)

# A100 GPU count for decode
A100_GPU_COUNT=8

# Port allocation for multiple decode servers
# Each decode server runs on a different GPU and port
DECODE_BASE_PORT=30000  # First decoder on 30000, second on 30001, etc.

# Prefill server (always on GH200)
PREFILL_HOST="${PREFILL_HOST:-172.16.40.79}"
PREFILL_PORT=30000

# A100 node for decoders
A100_HOST="${A100_HOST:-172.16.40.99}"

# Router port (runs on A100 node)
ROUTER_PORT="${ROUTER_PORT:-8000}"

# ===== HELPER FUNCTIONS =====

# Generate decode endpoints list for x decoders
# Usage: get_decode_endpoints 4  # Returns list of 4 decode URLs
get_decode_endpoints() {
    local num_decoders=$1
    local endpoints=()
    for ((i=0; i<num_decoders; i++)); do
        local port=$((DECODE_BASE_PORT + i))
        endpoints+=("http://127.0.0.1:${port}")
    done
    echo "${endpoints[@]}"
}

# Generate GPU list for x decoders
# Usage: get_gpu_list 4  # Returns "0,1,2,3"
get_gpu_list() {
    local num_gpus=$1
    local gpus=""
    for ((i=0; i<num_gpus; i++)); do
        if [ -n "$gpus" ]; then
            gpus="${gpus},"
        fi
        gpus="${gpus}${i}"
    done
    echo "$gpus"
}

# Estimate maximum concurrency based on model and memory
estimate_max_concurrency() {
    local model_size_gb=${1:-6}  # Default 6GB for Qwen2.5-3B
    local gpu_mem_gb=${2:-80}    # Default 80GB for A100
    local mem_fraction=${3:-0.9}
    local avg_seq_len=${4:-2048} # Average total sequence length
    
    # KV cache per token (approximate for 3B model)
    local kv_per_token_mb=0.05
    
    # Available memory for KV cache
    local available_gb=$(echo "$gpu_mem_gb * $mem_fraction - $model_size_gb - 2" | bc)
    local available_mb=$(echo "$available_gb * 1024" | bc)
    
    # Max tokens in KV cache
    local max_tokens=$(echo "$available_mb / $kv_per_token_mb" | bc)
    
    # Max concurrent requests
    local max_concurrent=$(echo "$max_tokens / $avg_seq_len" | bc)
    
    echo "$max_concurrent"
}

# Print configuration summary
print_extended_config() {
    echo "=============================================="
    echo "Extended Benchmark Configuration"
    echo "=============================================="
    echo "Model: ${MODEL_PATH}"
    echo "Context Length: ${MODEL_CONTEXT_LENGTH}"
    echo ""
    echo "Hardware:"
    echo "  Prefill: GH200 @ ${PREFILL_HOST}:${PREFILL_PORT}"
    echo "  Decode: A100 x${A100_GPU_COUNT} @ ${A100_HOST}:${DECODE_BASE_PORT}+"
    echo ""
    echo "Max Parameters:"
    echo "  Num Prompts: ${MAX_NUM_PROMPTS[*]}"
    echo "  Input Lengths: ${MAX_INPUT_LEN[*]}"
    echo "  Output Lengths: ${MAX_OUTPUT_LEN[*]}"
    echo "  Concurrency: ${MAX_CONCURRENCY[*]}"
    echo ""
    echo "Estimated Max Concurrency (A100 80GB):"
    echo "  Per GPU: ~$(estimate_max_concurrency 6 80 0.9 2048)"
    echo "  With 8 GPUs: ~$(($(estimate_max_concurrency 6 80 0.9 2048) * 8))"
    echo "=============================================="
}

# Export all variables
export MODEL_PATH MODEL_CONTEXT_LENGTH
export MAX_NUM_PROMPTS MAX_INPUT_LEN MAX_OUTPUT_LEN MAX_CONCURRENCY
export A100_GPU_COUNT DECODE_BASE_PORT PREFILL_HOST PREFILL_PORT
export A100_HOST ROUTER_PORT


