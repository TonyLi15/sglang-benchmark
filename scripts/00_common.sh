#!/usr/bin/env bash
set -euo pipefail

# ===== Common configuration (override via env if needed) =====

# HF model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

# SGLang Docker image (ARM/GH200)
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:dev-arm64}"

# Hugging Face cache dir (shared with containers)
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Cluster IPs (defaults to your current GH200 IPs)
PREFILL_HOST="${PREFILL_HOST:-172.16.40.79}"   # cg1n1
DECODE_HOST="${DECODE_HOST:-172.16.40.80}"     # cg1n2

# Ports
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30000}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

# Host venv for router + bench (on cg1n2)
VENV_DIR="${VENV_DIR:-$HOME/venv_sglang}"

# Benchmark settings
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-512}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-200}"

# Helper: repo root (assumes scripts/ is at top-level)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"
