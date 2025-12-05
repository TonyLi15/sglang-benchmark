#!/usr/bin/env bash
set -euo pipefail

# ===== Common configuration (override via env if needed) =====

# HF model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

# SGLang Docker images
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:dev-arm64}"  # GH200 (ARM)
SGLANG_IMAGE_X86="${SGLANG_IMAGE_X86:-lmsysorg/sglang:latest}"  # A100 (x86)

# Hugging Face cache dir (shared with containers)
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# ===== Node IPs =====
# GH200 nodes
PREFILL_HOST="${PREFILL_HOST:-172.16.40.79}"   # cg1n1 (GH200)
DECODE_HOST="${DECODE_HOST:-172.16.40.80}"     # cg1n2 (GH200)
# A100 node
A100_HOST="${A100_HOST:-172.16.40.99}"         # A100 node

# ===== Ports =====
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"  # Different port for intra-node
ROUTER_PORT="${ROUTER_PORT:-8000}"

# ===== RDMA Configuration =====
IB_DEVICE="${IB_DEVICE:-mlx5_0}"          # InfiniBand on GH200
A100_IB_DEVICE="${A100_IB_DEVICE:-mlx5_4}" # RoCE on A100

# ===== Transfer Backends =====
# mooncake: Same-fabric RDMA (IB-IB or RoCE-RoCE)
# nixl: Cross-fabric RDMA via UCX (IB-RoCE compatible)
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}"
INTER_NODE_TRANSFER_BACKEND="${INTER_NODE_TRANSFER_BACKEND:-nixl}"

# ===== Memory Configuration =====
MEM_FRACTION="${MEM_FRACTION:-0.45}"  # Lower for running both on same GPU

# ===== Host venv for router + bench =====
VENV_DIR="${VENV_DIR:-$HOME/venv_sglang}"

# ===== Benchmark settings =====
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-512}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-200}"

# ===== Helper paths =====
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"
