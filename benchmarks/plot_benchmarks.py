#!/usr/bin/env python3
"""
Plot benchmark comparison: Aggregated vs PD Disaggregation
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"

# Modern color palette
COLORS = {
    'agg': '#2E86AB',      # Blue for aggregated
    'pd': '#A23B72',       # Magenta for PD disaggregation
    'highlight': '#F18F01', # Orange for highlights
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'grid.alpha': 0.3,
})


def load_results():
    """Load all JSONL results from the results directory."""
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
                    "num_prompts": rec.get("num_prompts"),
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
                    "median_ttft_ms": rec.get("median_ttft_ms"),
                    "p99_ttft_ms": rec.get("p99_ttft_ms"),
                    "mean_tpot_ms": rec.get("mean_tpot_ms"),
                    "mean_itl_ms": rec.get("mean_itl_ms"),
                    "is_pd": "pd" in tag.lower() or "disagg" in tag.lower(),
                })
    if not rows:
        raise SystemExit(f"No JSONL results found in {RESULTS_DIR}")
    return pd.DataFrame(rows)


def plot_comparison(df):
    """Create a comprehensive comparison plot."""
    # Get latest result per tag
    df_last = df.sort_values("file").groupby("tag").tail(1).copy()
    
    # Separate aggregated and PD results
    df_agg = df_last[~df_last["is_pd"]]
    df_pd = df_last[df_last["is_pd"]]
    
    if df_agg.empty or df_pd.empty:
        print("Need both aggregated and PD results for comparison")
        return
    
    # Use the latest of each
    agg = df_agg.iloc[-1]
    pd_result = df_pd.iloc[-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SGLang Benchmark: Aggregated vs PD Disaggregation (Intra-Node)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    labels = ['Aggregated', 'PD Disagg']
    x = np.arange(len(labels))
    width = 0.6
    
    # 1. Throughput comparison
    ax1 = axes[0, 0]
    throughputs = [agg['total_throughput'], pd_result['total_throughput']]
    bars1 = ax1.bar(x, throughputs, width, color=[COLORS['agg'], COLORS['pd']])
    ax1.set_ylabel('Tokens/sec')
    ax1.set_title('Total Throughput')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.bar_label(bars1, fmt='%.0f', padding=3)
    ax1.set_ylim(0, max(throughputs) * 1.15)
    
    # 2. TTFT comparison
    ax2 = axes[0, 1]
    ttfts = [agg['mean_ttft_ms'], pd_result['mean_ttft_ms']]
    bars2 = ax2.bar(x, ttfts, width, color=[COLORS['agg'], COLORS['pd']])
    ax2.set_ylabel('Milliseconds')
    ax2.set_title('Mean Time to First Token (TTFT)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.bar_label(bars2, fmt='%.0f ms', padding=3)
    ax2.set_ylim(0, max(ttfts) * 1.15)
    
    # 3. E2E Latency comparison
    ax3 = axes[1, 0]
    e2e = [agg['mean_e2e_ms'], pd_result['mean_e2e_ms']]
    bars3 = ax3.bar(x, e2e, width, color=[COLORS['agg'], COLORS['pd']])
    ax3.set_ylabel('Milliseconds')
    ax3.set_title('Mean End-to-End Latency')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.bar_label(bars3, fmt='%.0f ms', padding=3)
    ax3.set_ylim(0, max(e2e) * 1.15)
    
    # 4. Summary metrics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Aggregated', 'PD Disagg', 'Diff'],
        ['Throughput (tok/s)', f'{agg["total_throughput"]:.0f}', 
         f'{pd_result["total_throughput"]:.0f}',
         f'{(pd_result["total_throughput"]/agg["total_throughput"]-1)*100:+.1f}%'],
        ['Mean TTFT (ms)', f'{agg["mean_ttft_ms"]:.0f}', 
         f'{pd_result["mean_ttft_ms"]:.0f}',
         f'{(pd_result["mean_ttft_ms"]/agg["mean_ttft_ms"]-1)*100:+.1f}%'],
        ['Mean E2E (ms)', f'{agg["mean_e2e_ms"]:.0f}', 
         f'{pd_result["mean_e2e_ms"]:.0f}',
         f'{(pd_result["mean_e2e_ms"]/agg["mean_e2e_ms"]-1)*100:+.1f}%'],
        ['Req Throughput', f'{agg["request_throughput"]:.1f}', 
         f'{pd_result["request_throughput"]:.1f}',
         f'{(pd_result["request_throughput"]/agg["request_throughput"]-1)*100:+.1f}%'],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.22, 0.22, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style data rows
    for i in range(1, 5):
        for j in range(4):
            if j == 3:  # Diff column
                val = table_data[i][j]
                if val.startswith('+'):
                    table[(i, j)].set_text_props(color='#D32F2F')  # Red for worse
                else:
                    table[(i, j)].set_text_props(color='#388E3C')  # Green for better
    
    ax4.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "benchmark_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {out_path}")
    
    return out_path


def plot_ttft_breakdown(df):
    """Create detailed TTFT breakdown plot."""
    df_last = df.sort_values("file").groupby("tag").tail(1).copy()
    
    df_agg = df_last[~df_last["is_pd"]]
    df_pd = df_last[df_last["is_pd"]]
    
    if df_agg.empty or df_pd.empty:
        return
    
    agg = df_agg.iloc[-1]
    pd_result = df_pd.iloc[-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Mean', 'Median', 'P99']
    x = np.arange(len(labels))
    width = 0.35
    
    agg_ttfts = [agg['mean_ttft_ms'], agg.get('median_ttft_ms', 0), agg.get('p99_ttft_ms', 0)]
    pd_ttfts = [pd_result['mean_ttft_ms'], pd_result.get('median_ttft_ms', 0), pd_result.get('p99_ttft_ms', 0)]
    
    bars1 = ax.bar(x - width/2, agg_ttfts, width, label='Aggregated', color=COLORS['agg'])
    bars2 = ax.bar(x + width/2, pd_ttfts, width, label='PD Disagg', color=COLORS['pd'])
    
    ax.set_ylabel('Milliseconds')
    ax.set_title('Time to First Token (TTFT) Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(bars1, fmt='%.0f', padding=3)
    ax.bar_label(bars2, fmt='%.0f', padding=3)
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "ttft_breakdown.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved TTFT breakdown: {out_path}")


def main():
    df = load_results()
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    cols = ["tag", "total_throughput", "mean_ttft_ms", "mean_e2e_ms", "request_throughput"]
    print(df[cols].to_string(index=False))
    print("="*60 + "\n")
    
    # Generate plots
    plot_comparison(df)
    plot_ttft_breakdown(df)
    
    print("\nPlots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
