#!/usr/bin/env bash
set -euo pipefail

# Get the repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Qwen/Qwen2.5-3B-Instruct supports up to 32k context length.
# Generation is capped at 8k.
# We adjust the sweep to test various ranges within these limits.
# Ensure input_len + output_len <= 32768.

NUM_PROMPTS_LIST=(200)
# Testing short, medium, and long contexts
INPUT_LEN_LIST=(512 4096 16384 28000)
OUTPUT_LEN_LIST=(128 2048)
CONCURRENCY_LIST=(128)

# Select benchmark mode: agg (default) or pd
MODE="${1:-agg}"

echo "Starting sweep for mode: ${MODE}"

for num_prompts in "${NUM_PROMPTS_LIST[@]}"; do
    for input_len in "${INPUT_LEN_LIST[@]}"; do
        for output_len in "${OUTPUT_LEN_LIST[@]}"; do
            for concurrency in "${CONCURRENCY_LIST[@]}"; do
                
                # Construct a unique tag for this run
                TAG="${MODE}_n${num_prompts}_in${input_len}_out${output_len}_conc${concurrency}"
                
                echo "========================================================"
                echo "Running sweep: ${TAG}"
                echo "  Num Prompts: ${num_prompts}"
                echo "  Input Len:   ${input_len}"
                echo "  Output Len:  ${output_len}"
                echo "  Concurrency: ${concurrency}"
                echo "========================================================"
                
                # Export variables to override defaults in scripts/00_common.sh
                export BENCH_NUM_PROMPTS="${num_prompts}"
                export BENCH_INPUT_LEN="${input_len}"
                export BENCH_OUTPUT_LEN="${output_len}"
                export BENCH_MAX_CONCURRENCY="${concurrency}"
                export TAG="${TAG}"
                
                # Run the appropriate benchmark script
                if [ "$MODE" == "agg" ]; then
                    bash "${REPO_ROOT}/scripts/23_bench_agg.sh"
                elif [ "$MODE" == "pd" ]; then
                    bash "${REPO_ROOT}/scripts/24_bench_pd_disagg.sh"
                else
                    echo "Unknown mode: ${MODE}. Use 'agg' or 'pd'."
                    exit 1
                fi
                
            done
        done
    done
done

echo "Sweep completed."

