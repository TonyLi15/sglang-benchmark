# SGLang Benchmark: Aggregated vs PD Disaggregation

This repository provides reproducible benchmarks comparing:

- **Aggregated** (prefill + decode on a single server)
- **Intra-Node PD Disaggregation** (prefill + decode as separate processes on same node)
- **Inter-Node PD Disaggregation** (prefill on GH200, decode on A100, connected via NIXL)

using **SGLang**, `Qwen/Qwen2.5-3B-Instruct`, and heterogeneous GPU clusters.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PD Disaggregation Modes                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Aggregated (Single Node)      Intra-Node PD        Inter-Node PD  │
│  ┌──────────────────┐         ┌──────────────┐     ┌─────────────┐ │
│  │  GH200           │         │  GH200       │     │  GH200      │ │
│  │ ┌──────────────┐ │         │ ┌──────────┐ │     │  (Prefill)  │ │
│  │ │   Prefill    │ │         │ │ Prefill  │ │     └──────┬──────┘ │
│  │ │      +       │ │         │ └────┬─────┘ │            │        │
│  │ │   Decode     │ │         │      │RDMA   │        NIXL/UCX     │
│  │ └──────────────┘ │         │ ┌────┴─────┐ │            │        │
│  └──────────────────┘         │ │ Decode   │ │     ┌──────┴──────┐ │
│                               │ └──────────┘ │     │    A100     │ │
│                               └──────────────┘     │   (Decode)  │ │
│                                                    └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```text
sglang-benchmark/
├── README.md
├── scripts/
│   ├── 00_common.sh               # Shared configuration
│   ├── 01_setup_host_venv.sh      # Host venv for router + benchmarks
│   │
│   │   # Aggregated (single server)
│   ├── 10_run_agg_server.sh       # Aggregated baseline server
│   │
│   │   # GH200 Inter-Node PD (Mooncake backend)
│   ├── 20_run_prefill_container.sh# Prefill-only server (cg1n1)
│   ├── 21_run_decode_container.sh # Decode-only server (cg1n2)
│   ├── 22_run_router_pd.sh        # Router for PD disaggregation
│   │
│   │   # Intra-Node PD (single node, 2 processes)
│   ├── 30_run_intra_node_pd.sh    # Start prefill & decode on same node
│   ├── 31_start_router.sh         # Start the MiniLB router
│   ├── 32_bench_intra_node_pd.sh  # Benchmark intra-node PD
│   ├── 33_full_intra_node_pd_bench.sh  # Full automated benchmark
│   │
│   │   # Inter-Node PD: GH200 + A100 (NIXL backend)
│   ├── 40_run_decode_a100.sh      # Decode server on A100
│   ├── 41_run_prefill_gh200.sh    # Prefill server on GH200
│   ├── 42_bench_inter_node_pd.sh  # Benchmark inter-node PD
│   │
│   │   # Benchmarks
│   ├── 23_bench_agg.sh            # Benchmark aggregated baseline
│   └── 24_bench_pd_disagg.sh      # Benchmark PD-disaggregated setup
│
├── benchmarks/
│   ├── plot_benchmarks.py         # Parse JSONL & generate plots
│   └── results/                   # Benchmark output JSONL + PNG
│
└── experiment/
    └── run_sweep.sh               # Parameter sweep experiments
```

---

## Prerequisites

### Hardware
- **GH200 node** (cg1n1): NVIDIA GH200 with InfiniBand
- **A100 node** (optional): NVIDIA A100 with RoCE for inter-node PD

### Software
- Docker + NVIDIA container runtime
- Python 3.12 for host venv
- **NIXL** (for inter-node PD with heterogeneous RDMA): `pip install nixl`

### Verify Setup
```bash
# Check Docker + GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi

# Check RDMA (optional, for intra-node)
ibstat | grep -A5 mlx5_0
```

---

## Quick Start

### 1. Setup Environment

```bash
cd sglang-benchmark
bash scripts/01_setup_host_venv.sh
source ~/venv_sglang/bin/activate

# For inter-node PD, also install NIXL
pip install nixl
```

### 2. Run Aggregated Benchmark

```bash
# Start server
bash scripts/10_run_agg_server.sh
sleep 45  # Wait for initialization

# Benchmark
TAG=agg_local bash scripts/23_bench_agg.sh

# Stop
docker stop sglang-agg
```

### 3. Run Intra-Node PD Benchmark

```bash
# Full automated (recommended)
TAG=pd_local bash scripts/33_full_intra_node_pd_bench.sh
```

### 4. Generate Plots

```bash
python3 benchmarks/plot_benchmarks.py
```

Output: `benchmarks/results/benchmark_comparison.png`

---

## Inter-Node PD Disaggregation (GH200 + A100)

### Why NIXL?

| Backend | Transport | Cross-Fabric Support |
|---------|-----------|---------------------|
| **Mooncake** | RDMA (IB/RoCE) | ❌ Same fabric only |
| **NIXL** | UCX (RDMA + TCP) | ✅ Works across IB ↔ RoCE |

When GH200 uses **InfiniBand** and A100 uses **RoCE**, they cannot communicate via Mooncake. **NIXL** solves this by using UCX which supports multiple transports.

### Setup Steps

#### On GH200 (Prefill Server)
```bash
# Install NIXL in venv
source ~/venv_sglang/bin/activate
pip install nixl

# Start prefill server
bash scripts/41_run_prefill_gh200.sh
```

#### On A100 (Decode Server + Router)
```bash
# Install NIXL
pip install nixl

# Start decode server
bash scripts/40_run_decode_a100.sh
sleep 60  # Wait for initialization

# Start router
source ~/venv_sglang/bin/activate
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://172.16.40.79:30000 \
  --decode http://127.0.0.1:30000 \
  --host 0.0.0.0 --port 8000

# Benchmark
TAG=pd_inter_node bash scripts/42_bench_inter_node_pd.sh
```

### Network Configuration

| Node | IP | RDMA Fabric | Device |
|------|-----|-------------|--------|
| GH200 (cg1n1) | 172.16.40.79 | InfiniBand | mlx5_0 |
| A100 | 172.16.40.99 | RoCE | mlx5_4 |

---

## Benchmark Results

Example results from our setup:

| Configuration | Throughput | Mean TTFT | Mean E2E |
|---------------|------------|-----------|----------|
| Aggregated (GH200) | 36,914 tok/s | 618 ms | 1,491 ms |
| PD Intra-Node (GH200) | 12,053 tok/s | 1,753 ms | 2,725 ms |
| PD Inter-Node (GH200→A100) | 1,209 tok/s | 282 ms | 564 ms |

**Key Insight**: Inter-node PD has **lower TTFT** because prefill is offloaded to GH200 while A100 handles decode with dedicated resources.

---

## Configuration Options

Edit `scripts/00_common.sh`:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | HuggingFace model | `Qwen/Qwen2.5-3B-Instruct` |
| `SGLANG_IMAGE` | Docker image | `lmsysorg/sglang:dev-arm64` |
| `PREFILL_PORT` | Prefill server port | `30000` |
| `DECODE_PORT` | Decode server port | `30001` |
| `ROUTER_PORT` | Router port | `8000` |
| `IB_DEVICE` | InfiniBand device | `mlx5_0` |
| `MEM_FRACTION` | GPU memory fraction | `0.45` |
| `BENCH_NUM_PROMPTS` | Number of prompts | `200` |
| `BENCH_INPUT_LEN` | Input token length | `512` |
| `BENCH_OUTPUT_LEN` | Output token length | `128` |
| `BENCH_MAX_CONCURRENCY` | Max concurrent requests | `200` |

---

## Troubleshooting

### TTFT shows 0 in PD disaggregation

The router needs to use `iter_any()` for streaming:
```python
# In sglang_router/mini_lb.py, generate_stream method:
async for chunk in decode_response.content.iter_any():  # NOT iter_chunked()
    yield chunk
```

### KVTransferError: Failed to get kvcache

**For same-fabric RDMA (Mooncake):**
1. Check `nvidia-peermem` is loaded: `lsmod | grep peermem`
2. Check IB device: `ibstat | grep -A10 mlx5_0`

**For cross-fabric (GH200 IB ↔ A100 RoCE):**
- Use NIXL backend: `--disaggregation-transfer-backend nixl`

### Container fails to start

Check GPU memory: `nvidia-smi`
Try reducing `MEM_FRACTION` in `00_common.sh`

### Inter-node PD not working between different RDMA fabrics

```bash
# Check RDMA type on each node
ibstat | grep "Link layer"

# If different (InfiniBand vs Ethernet), use NIXL:
--disaggregation-transfer-backend nixl
```

---

## References

- [SGLang PD Disaggregation Docs](https://docs.sglang.io/advanced_features/pd_disaggregation.html)
- [NIXL GitHub](https://github.com/ai-dynamo/nixl)
- [Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake)

---

## License

MIT - Feel free to adapt for your own experiments.
