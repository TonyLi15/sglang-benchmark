#!/usr/bin/env python3
"""
Plot benchmark comparison: Aggregated vs PD Disaggregation
Supports both single results and parameter sweep results.
"""
import json
import pathlib
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"

# Modern color palette
COLORS = {
    'agg': '#2E86AB',           # Blue for aggregated
    'pd_intra': '#A23B72',      # Magenta for intra-node PD
    'pd_inter': '#F18F01',      # Orange for inter-node PD
    'highlight': '#28A745',     # Green for highlights
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
                
                # Parse sweep parameters from tag if present
                # Format: mode_nX_inY_outZ_cW
                match = re.match(r'(\w+)_n(\d+)_in(\d+)_out(\d+)_c(\d+)', tag)
                if match:
                    mode = match.group(1)
                    sweep_params = {
                        'num_prompts': int(match.group(2)),
                        'input_len': int(match.group(3)),
                        'output_len': int(match.group(4)),
                        'concurrency': int(match.group(5)),
                    }
                else:
                    # Determine mode from tag
                    if 'inter' in tag.lower():
                        mode = 'pd_inter'
                    elif 'pd' in tag.lower() or 'disagg' in tag.lower():
                        mode = 'pd_intra'
                    else:
                        mode = 'agg'
                    sweep_params = {
                        'num_prompts': rec.get("num_prompts", 0),
                        'input_len': rec.get("random_input_len", 0),
                        'output_len': rec.get("random_output_len", 0),
                        'concurrency': rec.get("max_concurrency", 0),
                    }
                
                rows.append({
                    "tag": tag,
                    "mode": mode,
                    "file": path.name,
                    "backend": rec.get("backend"),
                    "dataset_name": rec.get("dataset_name"),
                    **sweep_params,
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
                })
    if not rows:
        raise SystemExit(f"No JSONL results found in {RESULTS_DIR}")
    return pd.DataFrame(rows)


def get_mode_label(mode):
    """Get display label for mode."""
    labels = {
        'agg': 'Aggregated',
        'pd_intra': 'Intra-Node PD',
        'pd_inter': 'Inter-Node PD',
    }
    return labels.get(mode, mode)


def plot_comparison(df):
    """Create a comprehensive comparison plot."""
    # Get latest result per mode (for simple comparison)
    df_by_mode = df.groupby("mode").last().reset_index()
    
    if len(df_by_mode) < 2:
        print("Need at least 2 modes for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('SGLang Benchmark: Aggregated vs PD Disaggregation', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    modes = df_by_mode['mode'].tolist()
    labels = [get_mode_label(m) for m in modes]
    colors = [COLORS.get(m, '#999999') for m in modes]
    x = np.arange(len(labels))
    width = 0.6
    
    # 1. Throughput comparison
    ax1 = axes[0, 0]
    throughputs = df_by_mode['total_throughput'].tolist()
    bars1 = ax1.bar(x, throughputs, width, color=colors)
    ax1.set_ylabel('Tokens/sec')
    ax1.set_title('Total Throughput')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.bar_label(bars1, fmt='%.0f', padding=3)
    ax1.set_ylim(0, max(throughputs) * 1.2)
    
    # 2. TTFT comparison
    ax2 = axes[0, 1]
    ttfts = df_by_mode['mean_ttft_ms'].tolist()
    bars2 = ax2.bar(x, ttfts, width, color=colors)
    ax2.set_ylabel('Milliseconds')
    ax2.set_title('Mean Time to First Token (TTFT)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.bar_label(bars2, fmt='%.0f ms', padding=3)
    ax2.set_ylim(0, max(ttfts) * 1.2)
    
    # 3. E2E Latency comparison
    ax3 = axes[1, 0]
    e2e = df_by_mode['mean_e2e_ms'].tolist()
    bars3 = ax3.bar(x, e2e, width, color=colors)
    ax3.set_ylabel('Milliseconds')
    ax3.set_title('Mean End-to-End Latency')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha='right')
    ax3.bar_label(bars3, fmt='%.0f ms', padding=3)
    ax3.set_ylim(0, max(e2e) * 1.2)
    
    # 4. Summary metrics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Build table data
    header = ['Metric'] + labels
    table_data = [header]
    
    # Add metrics rows
    baseline_throughput = throughputs[0] if throughputs else 1
    baseline_ttft = ttfts[0] if ttfts else 1
    baseline_e2e = e2e[0] if e2e else 1
    
    table_data.append(['Throughput (tok/s)'] + [f'{t:.0f}' for t in throughputs])
    table_data.append(['Mean TTFT (ms)'] + [f'{t:.0f}' for t in ttfts])
    table_data.append(['Mean E2E (ms)'] + [f'{t:.0f}' for t in e2e])
    table_data.append(['Req Throughput'] + [f'{r:.1f}' for r in df_by_mode['request_throughput'].tolist()])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25] + [0.22] * len(labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header row
    for j in range(len(header)):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "benchmark_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {out_path}")
    
    return out_path


def plot_ttft_breakdown(df):
    """Create detailed TTFT breakdown plot."""
    df_by_mode = df.groupby("mode").last().reset_index()
    
    if len(df_by_mode) < 2:
        return
    
    modes = df_by_mode['mode'].tolist()
    labels = [get_mode_label(m) for m in modes]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Mean', 'Median', 'P99']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (mode, label) in enumerate(zip(modes, labels)):
        row = df_by_mode[df_by_mode['mode'] == mode].iloc[0]
        values = [
            row.get('mean_ttft_ms', 0) or 0,
            row.get('median_ttft_ms', 0) or 0,
            row.get('p99_ttft_ms', 0) or 0,
        ]
        offset = (i - len(modes)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=COLORS.get(mode, '#999999'))
        ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8)
    
    ax.set_ylabel('Milliseconds')
    ax.set_title('Time to First Token (TTFT) Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "ttft_breakdown.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved TTFT breakdown: {out_path}")


def plot_sweep_results(df):
    """Plot sweep results if multiple parameter combinations exist."""
    # Check if we have sweep results
    unique_params = df.groupby(['input_len', 'output_len', 'concurrency']).size()
    if len(unique_params) <= 1:
        return  # No sweep data
    
    # Plot throughput vs input length
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Parameter Sweep Results', fontsize=16, fontweight='bold', y=0.98)
    
    modes = df['mode'].unique()
    
    # 1. Throughput vs Input Length
    ax1 = axes[0, 0]
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        grouped = mode_df.groupby('input_len')['total_throughput'].mean()
        ax1.plot(grouped.index, grouped.values, 'o-', 
                label=get_mode_label(mode), color=COLORS.get(mode, '#999999'), linewidth=2)
    ax1.set_xlabel('Input Length (tokens)')
    ax1.set_ylabel('Throughput (tok/s)')
    ax1.set_title('Throughput vs Input Length')
    ax1.legend()
    ax1.set_xscale('log')
    
    # 2. TTFT vs Input Length
    ax2 = axes[0, 1]
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        grouped = mode_df.groupby('input_len')['mean_ttft_ms'].mean()
        ax2.plot(grouped.index, grouped.values, 'o-', 
                label=get_mode_label(mode), color=COLORS.get(mode, '#999999'), linewidth=2)
    ax2.set_xlabel('Input Length (tokens)')
    ax2.set_ylabel('TTFT (ms)')
    ax2.set_title('TTFT vs Input Length')
    ax2.legend()
    ax2.set_xscale('log')
    
    # 3. Throughput vs Concurrency
    ax3 = axes[1, 0]
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        grouped = mode_df.groupby('concurrency')['total_throughput'].mean()
        ax3.plot(grouped.index, grouped.values, 'o-', 
                label=get_mode_label(mode), color=COLORS.get(mode, '#999999'), linewidth=2)
    ax3.set_xlabel('Concurrency')
    ax3.set_ylabel('Throughput (tok/s)')
    ax3.set_title('Throughput vs Concurrency')
    ax3.legend()
    
    # 4. E2E Latency vs Concurrency
    ax4 = axes[1, 1]
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        grouped = mode_df.groupby('concurrency')['mean_e2e_ms'].mean()
        ax4.plot(grouped.index, grouped.values, 'o-', 
                label=get_mode_label(mode), color=COLORS.get(mode, '#999999'), linewidth=2)
    ax4.set_xlabel('Concurrency')
    ax4.set_ylabel('E2E Latency (ms)')
    ax4.set_title('E2E Latency vs Concurrency')
    ax4.legend()
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "sweep_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved sweep plot: {out_path}")


def main():
    df = load_results()
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    cols = ["tag", "total_throughput", "mean_ttft_ms", "mean_e2e_ms", "request_throughput"]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string(index=False))
    print("="*60 + "\n")
    
    # Generate plots
    plot_comparison(df)
    plot_ttft_breakdown(df)
    plot_sweep_results(df)
    
    print("\nPlots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
