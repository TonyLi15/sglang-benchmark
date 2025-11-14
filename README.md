# SGLang Benchmark (Aggregated vs PD Disaggregation on GH200)

This repository captures the setup and benchmark workflow for:

- **Aggregated PD** (prefill + decode on a single GH200 node)
- **PD Disaggregation** (prefill on `cg1n1`, decode on `cg1n2`, with a router)

using **SGLang**, `Qwen/Qwen2.5-3B-Instruct`, and **two GH200 servers**.

The goal is to make everything reproducible with shell scripts:
environment setup, server launch, PD router, benchmarks, and plotting.

---

## 0. Repository layout

```text
sglang-benchmark/
  README.md
  scripts/
    00_common.sh               # Shared configuration (IPs, ports, model, etc.)
    01_setup_host_venv.sh      # Host venv on cg1n2 for router + benchmarks

    10_run_agg_server.sh       # Aggregated baseline server (single node)
    20_run_prefill_container.sh# Prefill-only SGLang server (PD)
    21_run_decode_container.sh # Decode-only SGLang server (PD)
    22_run_router_pd.sh        # sglang-router (MiniLB) for PD disaggregation

    23_bench_agg.sh            # Benchmark aggregated baseline
    24_bench_pd_disagg.sh      # Benchmark PD-disaggregated setup

  benchmarks/
    plot_benchmarks.py         # Parse JSONL & plot throughput / latency
    results/
      .gitkeep                 # Benchmark output JSONL + PNG go here
```

You can adapt IPs, ports, model, and concurrency in `scripts/00_common.sh`.

---

## 1. Prerequisites

- Two GH200 nodes (here referred to as **`cg1n1`** and **`cg1n2`**)
- Working NVIDIA drivers + `nvidia-smi` on both
- Docker + NVIDIA container runtime on both:
  - `docker --version`
  - `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi`
- Python 3.12 on **cg1n2** (for host venv / router / plotting)

The default IPs used in this repo (editable in `00_common.sh`):

- `cg1n1`: `172.16.40.79`
- `cg1n2`: `172.16.40.80`

Model & image defaults:

- Model: `Qwen/Qwen2.5-3B-Instruct`
- SGLang image: `lmsysorg/sglang:dev-arm64`

---

## 2. Host venv for router + benchmarks (cg1n2)

On **cg1n2**:

```bash
cd sglang-benchmark

bash scripts/01_setup_host_venv.sh
source ~/venv_sglang/bin/activate
```

This creates a venv at `~/venv_sglang` and installs:

- `sglang` (client + bench)
- `sglang-router`
- `triton` (for benchmarking)
- `matplotlib`, `pandas` (for plotting)

---

## 3. Aggregated baseline (single-node)

### 3.1 Start aggregated server on cg1n1

On **cg1n1**:

```bash
cd sglang-benchmark
bash scripts/10_run_agg_server.sh
```

This runs a Docker container exposing:

- Aggregated SGLang server at `http://cg1n1:30000`

Check it:

```bash
curl http://localhost:30000/get_model_info
```

### 3.2 Benchmark aggregated server from cg1n2

On **cg1n2**:

```bash
cd sglang-benchmark
source ~/venv_sglang/bin/activate

TAG=agg_cg1n1 bash scripts/23_bench_agg.sh
```

This will:

- Send random traffic via `bench_serving` to `cg1n1:30000`
- Save metrics as JSONL under `benchmarks/results/agg_cg1n1.jsonl`

You can adjust concurrency, token lengths, etc. in `scripts/00_common.sh`.

---

## 4. PD Disaggregation (1P1D across two nodes)

### 4.1 Prefill server on cg1n1

On **cg1n1**:

```bash
cd sglang-benchmark
bash scripts/20_run_prefill_container.sh
```

This runs:

- Prefill-only SGLang server on `cg1n1:30000` with `--disaggregation-mode prefill`.

### 4.2 Decode server on cg1n2

On **cg1n2**:

```bash
cd sglang-benchmark
bash scripts/21_run_decode_container.sh
```

This runs:

- Decode-only SGLang server on `cg1n2:30000` with `--disaggregation-mode decode`.

### 4.3 PD router on cg1n2 (MiniLB via sglang-router)

On **cg1n2**, in a dedicated terminal:

```bash
cd sglang-benchmark
bash scripts/22_run_router_pd.sh
```

This uses `sglang_router.launch_router` with:

- `--mini-lb` and `--pd-disaggregation`
- `--prefill http://172.16.40.79:30000` (default `PREFILL_HOST:PREFILL_PORT`)
- `--decode  http://127.0.0.1:30000` (local decode)
- `--host 0.0.0.0 --port 8000`

Test from cg1n2:

```bash
curl http://127.0.0.1:8000/get_model_info
```

If it returns model info JSON → PD router is working.

---

## 5. Benchmark PD-disaggregated setup

On **cg1n2**, with router running and venv active:

```bash
cd sglang-benchmark
source ~/venv_sglang/bin/activate

TAG=pd_cg1n1_cg1n2 bash scripts/24_bench_pd_disagg.sh
```

This uses `bench_serving` with:

- `--pd-separated`
- `--base-url http://127.0.0.1:8000` (router)
- Random synthetic prompts with configurable input/output lengths & concurrency
- Saves results as `benchmarks/results/pd_cg1n1_cg1n2.jsonl`

Make sure `BENCH_*` parameters in `00_common.sh` match those used for aggregated baseline.

---

## 6. Plotting benchmark results

Once you have at least two runs (e.g. `agg_cg1n1.jsonl` and `pd_cg1n1_cg1n2.jsonl`), run:

```bash
cd sglang-benchmark
source ~/venv_sglang/bin/activate

python benchmarks/plot_benchmarks.py
```

This will:

- Load all JSONL results under `benchmarks/results/`
- Print a small summary table (tag, throughput, mean E2E, mean TTFT)
- Generate:

  - `benchmarks/results/throughput_by_tag.png`
  - `benchmarks/results/e2e_latency_by_tag.png`

You can then directly use these plots in reports or slides.

---

## 7. Notes / Tips

- If you change model or image, update `scripts/00_common.sh` and re-run the appropriate scripts.
- You can preserve your old aggregated containers by renaming them (e.g. `sglang-cg1n1-agg`) and starting new ones for PD.
- For heavier benchmarks, increase:
  - `BENCH_NUM_PROMPTS`
  - `BENCH_INPUT_LEN`
  - `BENCH_OUTPUT_LEN`
  - `BENCH_MAX_CONCURRENCY`

But monitor GPU memory with `nvidia-smi` to avoid OOM, especially if multiple containers or experiments run on the same node.

---

## 8. License

Feel free to adapt this repo structure and scripts for your own experiments.
