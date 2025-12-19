#!/usr/bin/env python3
"""
Plot 1PxD Scaling Analysis: Performance vs Number of Decoders

Analyzes how throughput, latency, and efficiency scale as we add
more decode servers (1P1D → 1P2D → 1P4D → 1P8D).
"""

import json
import glob
import os
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = RESULTS_DIR

def load_benchmark_results(pattern="pd_*1p*d_*.jsonl"):
    """Load all 1PxD benchmark results."""
    results = []
    
    for filepath in glob.glob(str(RESULTS_DIR / pattern)):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Extract configuration from tag
            tag = data.get('tag', os.path.basename(filepath))
            
            # Parse tag: pd_inter_1pXd_nN_inI_outO_cC or pd_1pXd_...
            match = re.match(
                r'pd_(?:inter_)?1p(\d+)d_n(\d+)_in(\d+)_out(\d+)_c(\d+)',
                tag
            )
            
            if match:
                num_decoders = int(match.group(1))
                num_prompts = int(match.group(2))
                input_len = int(match.group(3))
                output_len = int(match.group(4))
                concurrency = int(match.group(5))
            else:
                # Try simpler pattern
                match = re.search(r'1p(\d+)d', tag)
                if match:
                    num_decoders = int(match.group(1))
                else:
                    continue
                num_prompts = data.get('num_prompts', 0)
                input_len = data.get('random_input_len', 0)
                output_len = data.get('random_output_len', 0)
                concurrency = data.get('max_concurrency', 0)
            
            results.append({
                'file': filepath,
                'tag': tag,
                'num_decoders': num_decoders,
                'num_prompts': num_prompts,
                'input_len': input_len,
                'output_len': output_len,
                'concurrency': concurrency,
                'throughput': data.get('output_throughput', 0),
                'total_throughput': data.get('total_throughput', 0),
                'mean_ttft': data.get('mean_ttft_ms', 0),
                'mean_e2e': data.get('mean_e2e_latency_ms', 0),
                'p99_e2e': data.get('p99_e2e_latency_ms', 0),
                'mean_tpot': data.get('mean_tpot_ms', 0),
                'mean_itl': data.get('mean_itl_ms', 0),
            })
            
        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")
            continue
    
    return results


def plot_scaling_by_decoders(results, config_filter=None):
    """
    Plot scaling curves: performance metrics vs number of decoders.
    
    Args:
        results: List of benchmark results
        config_filter: Optional dict to filter by config (e.g., {'input_len': 1024})
    """
    # Filter results if needed
    if config_filter:
        results = [r for r in results if all(
            r.get(k) == v for k, v in config_filter.items()
        )]
    
    if not results:
        print("No matching results found")
        return
    
    # Group by number of decoders
    by_decoders = defaultdict(list)
    for r in results:
        by_decoders[r['num_decoders']].append(r)
    
    # Sort decoder counts
    decoder_counts = sorted(by_decoders.keys())
    
    if len(decoder_counts) < 2:
        print("Need at least 2 different decoder counts for scaling analysis")
        return
    
    # Aggregate metrics (mean across configs)
    throughputs = []
    ttfts = []
    e2es = []
    tpots = []
    
    for d in decoder_counts:
        runs = by_decoders[d]
        throughputs.append(np.mean([r['throughput'] for r in runs]))
        ttfts.append(np.mean([r['mean_ttft'] for r in runs]))
        e2es.append(np.mean([r['mean_e2e'] for r in runs]))
        tpots.append(np.mean([r['mean_tpot'] for r in runs]))
    
    # Calculate scaling efficiency
    base_throughput = throughputs[0]  # 1P1D baseline
    ideal_scaling = [base_throughput * d / decoder_counts[0] for d in decoder_counts]
    efficiency = [t / i * 100 if i > 0 else 0 for t, i in zip(throughputs, ideal_scaling)]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('1PxD Scaling Analysis: GH200 (Prefill) → A100 x[1-8] (Decode)\n'
                 f'Model: Qwen2.5-3B | Transfer: NIXL',
                 fontsize=14, fontweight='bold', y=0.98)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(decoder_counts)))
    
    # 1. Throughput vs Decoders
    ax1 = axes[0, 0]
    ax1.bar(decoder_counts, throughputs, color='#3498db', edgecolor='black', linewidth=1.2)
    ax1.plot(decoder_counts, ideal_scaling, 'r--', linewidth=2, marker='o', 
             label='Ideal Linear Scaling', markersize=8)
    ax1.set_xlabel('Number of Decode GPUs', fontsize=11)
    ax1.set_ylabel('Output Throughput (tok/s)', fontsize=11)
    ax1.set_title('Throughput Scaling', fontsize=12, fontweight='bold')
    ax1.set_xticks(decoder_counts)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (d, t) in enumerate(zip(decoder_counts, throughputs)):
        ax1.annotate(f'{t:.0f}', xy=(d, t), ha='center', va='bottom', 
                     fontsize=10, fontweight='bold')
    
    # 2. Scaling Efficiency
    ax2 = axes[0, 1]
    bars = ax2.bar(decoder_counts, efficiency, color='#2ecc71', edgecolor='black', linewidth=1.2)
    ax2.axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% Efficiency')
    ax2.set_xlabel('Number of Decode GPUs', fontsize=11)
    ax2.set_ylabel('Scaling Efficiency (%)', fontsize=11)
    ax2.set_title('Scaling Efficiency vs Ideal', fontsize=12, fontweight='bold')
    ax2.set_xticks(decoder_counts)
    ax2.set_ylim(0, 120)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for d, e in zip(decoder_counts, efficiency):
        ax2.annotate(f'{e:.1f}%', xy=(d, e), ha='center', va='bottom', 
                     fontsize=10, fontweight='bold')
    
    # 3. TTFT vs Decoders
    ax3 = axes[0, 2]
    ax3.bar(decoder_counts, ttfts, color='#e74c3c', edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Number of Decode GPUs', fontsize=11)
    ax3.set_ylabel('Mean TTFT (ms)', fontsize=11)
    ax3.set_title('Time To First Token', fontsize=12, fontweight='bold')
    ax3.set_xticks(decoder_counts)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for d, t in zip(decoder_counts, ttfts):
        ax3.annotate(f'{t:.0f}', xy=(d, t), ha='center', va='bottom', 
                     fontsize=10, fontweight='bold')
    
    # 4. E2E Latency vs Decoders
    ax4 = axes[1, 0]
    ax4.bar(decoder_counts, e2es, color='#9b59b6', edgecolor='black', linewidth=1.2)
    ax4.set_xlabel('Number of Decode GPUs', fontsize=11)
    ax4.set_ylabel('Mean E2E Latency (ms)', fontsize=11)
    ax4.set_title('End-to-End Latency', fontsize=12, fontweight='bold')
    ax4.set_xticks(decoder_counts)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for d, e in zip(decoder_counts, e2es):
        ax4.annotate(f'{e:.0f}', xy=(d, e), ha='center', va='bottom', 
                     fontsize=10, fontweight='bold')
    
    # 5. Time Per Output Token vs Decoders
    ax5 = axes[1, 1]
    ax5.bar(decoder_counts, tpots, color='#f39c12', edgecolor='black', linewidth=1.2)
    ax5.set_xlabel('Number of Decode GPUs', fontsize=11)
    ax5.set_ylabel('Mean TPOT (ms)', fontsize=11)
    ax5.set_title('Time Per Output Token', fontsize=12, fontweight='bold')
    ax5.set_xticks(decoder_counts)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    for d, t in zip(decoder_counts, tpots):
        ax5.annotate(f'{t:.2f}', xy=(d, t), ha='center', va='bottom', 
                     fontsize=10, fontweight='bold')
    
    # 6. Throughput per GPU (efficiency metric)
    throughput_per_gpu = [t / d for t, d in zip(throughputs, decoder_counts)]
    ax6 = axes[1, 2]
    ax6.bar(decoder_counts, throughput_per_gpu, color='#1abc9c', edgecolor='black', linewidth=1.2)
    ax6.axhline(y=throughputs[0], color='r', linestyle='--', linewidth=2, 
                label=f'1P1D baseline ({throughputs[0]:.0f})')
    ax6.set_xlabel('Number of Decode GPUs', fontsize=11)
    ax6.set_ylabel('Throughput per GPU (tok/s)', fontsize=11)
    ax6.set_title('Per-GPU Efficiency', fontsize=12, fontweight='bold')
    ax6.set_xticks(decoder_counts)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    for d, t in zip(decoder_counts, throughput_per_gpu):
        ax6.annotate(f'{t:.0f}', xy=(d, t), ha='center', va='bottom', 
                     fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot
    output_file = OUTPUT_DIR / "1pxd_scaling_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Scaling plot saved to: {output_file}")
    
    return fig


def plot_heatmap_throughput(results):
    """
    Create heatmap of throughput across different configurations.
    X-axis: input_len, Y-axis: num_decoders, Color: throughput
    """
    # Get unique values
    decoder_counts = sorted(set(r['num_decoders'] for r in results))
    input_lens = sorted(set(r['input_len'] for r in results))
    
    if len(decoder_counts) < 2 or len(input_lens) < 2:
        print("Not enough data for heatmap")
        return
    
    # Build heatmap matrix
    throughput_matrix = np.zeros((len(decoder_counts), len(input_lens)))
    
    for r in results:
        d_idx = decoder_counts.index(r['num_decoders'])
        i_idx = input_lens.index(r['input_len'])
        throughput_matrix[d_idx, i_idx] = max(
            throughput_matrix[d_idx, i_idx],
            r['throughput']
        )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(throughput_matrix, cmap='YlOrRd', aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(input_lens)))
    ax.set_xticklabels(input_lens)
    ax.set_yticks(range(len(decoder_counts)))
    ax.set_yticklabels([f'1P{d}D' for d in decoder_counts])
    
    ax.set_xlabel('Input Length (tokens)', fontsize=12)
    ax.set_ylabel('Configuration', fontsize=12)
    ax.set_title('Throughput Heatmap: 1PxD Scaling × Input Length\n'
                 'GH200 (Prefill) → A100 (Decode)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Output Throughput (tok/s)', fontsize=11)
    
    # Add text annotations
    for i in range(len(decoder_counts)):
        for j in range(len(input_lens)):
            value = throughput_matrix[i, j]
            if value > 0:
                text_color = 'white' if value > throughput_matrix.max() * 0.6 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                       color=text_color, fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "1pxd_throughput_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved to: {output_file}")
    
    return fig


def print_summary_table(results):
    """Print summary table of results."""
    print("\n" + "=" * 100)
    print("1PxD SCALING BENCHMARK RESULTS")
    print("=" * 100)
    
    # Group by decoder count
    by_decoders = defaultdict(list)
    for r in results:
        by_decoders[r['num_decoders']].append(r)
    
    print(f"\n{'Config':<12} {'Throughput':>12} {'Mean TTFT':>12} {'Mean E2E':>12} "
          f"{'P99 E2E':>12} {'TPOT':>10} {'Runs':>6}")
    print(f"{'':12} {'(tok/s)':>12} {'(ms)':>12} {'(ms)':>12} "
          f"{'(ms)':>12} {'(ms)':>10} {'':>6}")
    print("-" * 100)
    
    for d in sorted(by_decoders.keys()):
        runs = by_decoders[d]
        avg_tp = np.mean([r['throughput'] for r in runs])
        avg_ttft = np.mean([r['mean_ttft'] for r in runs])
        avg_e2e = np.mean([r['mean_e2e'] for r in runs])
        avg_p99 = np.mean([r['p99_e2e'] for r in runs])
        avg_tpot = np.mean([r['mean_tpot'] for r in runs])
        
        print(f"1P{d}D{'':<8} {avg_tp:>12.2f} {avg_ttft:>12.2f} {avg_e2e:>12.2f} "
              f"{avg_p99:>12.2f} {avg_tpot:>10.2f} {len(runs):>6}")
    
    print("=" * 100)
    
    # Calculate speedup from 1P1D
    if 1 in by_decoders:
        base_tp = np.mean([r['throughput'] for r in by_decoders[1]])
        print("\nSpeedup vs 1P1D:")
        for d in sorted(by_decoders.keys()):
            if d == 1:
                continue
            runs = by_decoders[d]
            avg_tp = np.mean([r['throughput'] for r in runs])
            speedup = avg_tp / base_tp if base_tp > 0 else 0
            efficiency = speedup / d * 100
            print(f"  1P{d}D: {speedup:.2f}x speedup ({efficiency:.1f}% efficiency)")


def main():
    """Main entry point."""
    print("Loading 1PxD benchmark results...")
    
    # Load results
    results = load_benchmark_results()
    
    if not results:
        print("\nNo 1PxD benchmark results found.")
        print("Run benchmarks first:")
        print("  bash experiment/run_1pxd_sweep.sh")
        return
    
    print(f"Found {len(results)} benchmark result(s)")
    
    # Print summary
    print_summary_table(results)
    
    # Generate plots
    print("\nGenerating plots...")
    
    try:
        plot_scaling_by_decoders(results)
    except Exception as e:
        print(f"Warning: Could not generate scaling plot: {e}")
    
    try:
        plot_heatmap_throughput(results)
    except Exception as e:
        print(f"Warning: Could not generate heatmap: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

