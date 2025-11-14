#!/usr/bin/env python3
import json
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"


def load_results():
    rows = []
    for path in RESULTS_DIR.glob("*.jsonl"):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tag = rec.get("tag") or path.stem
                rows.append({
                    "tag": tag,
                    "file": path.name,
                    "backend": rec.get("backend"),
                    "dataset_name": rec.get("dataset_name"),
                    "request_rate": rec.get("request_rate"),
                    "max_concurrency": rec.get("max_concurrency"),
                    "duration_s": rec.get("duration"),
                    "request_throughput": rec.get("request_throughput"),
                    "input_throughput": rec.get("input_throughput"),
                    "output_throughput": rec.get("output_throughput"),
                    "total_throughput": rec.get("total_throughput"),
                    "mean_e2e_ms": rec.get("mean_e2e_latency_ms"),
                    "median_e2e_ms": rec.get("median_e2e_latency_ms"),
                    "mean_ttft_ms": rec.get("mean_ttft_ms"),
                    "mean_itl_ms": rec.get("mean_itl_ms"),
                })
    if not rows:
        raise SystemExit(f"No JSONL results found in {RESULTS_DIR}")
    return pd.DataFrame(rows)


def main():
    df = load_results()
    print("Loaded results:")
    print(df[["tag", "file", "request_throughput",
              "total_throughput", "mean_e2e_ms", "mean_ttft_ms"]])

    # If multiple entries per tag, keep the last (by file name order)
    df_last = df.sort_values("file").groupby("tag").tail(1)

    # Bar plot: total token throughput vs tag
    plt.figure()
    plt.bar(df_last["tag"], df_last["total_throughput"])
    plt.ylabel("Total token throughput (tok/s)")
    plt.xlabel("Setup tag")
    plt.title("SGLang Throughput: Aggregated vs PD Disaggregation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out1 = RESULTS_DIR / "throughput_by_tag.png"
    plt.savefig(out1)
    print(f"Saved {out1}")

    # Bar plot: mean E2E latency vs tag
    plt.figure()
    plt.bar(df_last["tag"], df_last["mean_e2e_ms"])
    plt.ylabel("Mean E2E latency (ms)")
    plt.xlabel("Setup tag")
    plt.title("SGLang Mean E2E Latency")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out2 = RESULTS_DIR / "e2e_latency_by_tag.png"
    plt.savefig(out2)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
