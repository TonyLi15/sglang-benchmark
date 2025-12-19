#!/usr/bin/env python3
"""
Plot benchmark comparison for maximum configuration:
n=100, input=512, output=128, concurrency=32
"""

import matplotlib.pyplot as plt
import numpy as np

# Data for max configuration: n100_in512_out128_c32
configs = ['Aggregated\n(GH200)', 'Intra-Node PD\n(GH200)', 'Inter-Node PD\n(GH200→A100)']
short_configs = ['Aggregated', 'Intra-Node PD', 'Inter-Node PD']

# Metrics extracted from jsonl files
throughput = [3151.44, 4211.80, 662.85]  # output tok/s
mean_ttft = [42.45, 80.06, 2345.04]  # ms
mean_e2e = [557.91, 388.63, 2732.99]  # ms
p99_e2e = [1181.29, 738.49, 3694.76]  # ms

# Colors - distinctive palette
colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SGLang PD Disaggregation Benchmark\nMax Config: input=512, output=128, concurrency=32, n=100', 
             fontsize=14, fontweight='bold', y=0.98)

# 1. Throughput (top-left)
ax1 = axes[0, 0]
bars1 = ax1.bar(short_configs, throughput, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Output Throughput (tok/s)', fontsize=11)
ax1.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(throughput) * 1.2)
for bar, val in zip(bars1, throughput):
    ax1.annotate(f'{val:.0f}', 
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 2. Mean TTFT (top-right)
ax2 = axes[0, 1]
bars2 = ax2.bar(short_configs, mean_ttft, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Mean TTFT (ms)', fontsize=11)
ax2.set_title('Time To First Token (TTFT)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(mean_ttft) * 1.2)
for bar, val in zip(bars2, mean_ttft):
    ax2.annotate(f'{val:.1f}', 
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 3. Mean E2E Latency (bottom-left)
ax3 = axes[1, 0]
bars3 = ax3.bar(short_configs, mean_e2e, color=colors, edgecolor='black', linewidth=1.2)
ax3.set_ylabel('Mean E2E Latency (ms)', fontsize=11)
ax3.set_title('End-to-End Latency (Mean)', fontsize=12, fontweight='bold')
ax3.set_ylim(0, max(mean_e2e) * 1.2)
for bar, val in zip(bars3, mean_e2e):
    ax3.annotate(f'{val:.0f}', 
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 4. P99 E2E Latency (bottom-right)
ax4 = axes[1, 1]
bars4 = ax4.bar(short_configs, p99_e2e, color=colors, edgecolor='black', linewidth=1.2)
ax4.set_ylabel('P99 E2E Latency (ms)', fontsize=11)
ax4.set_title('End-to-End Latency (P99)', fontsize=12, fontweight='bold')
ax4.set_ylim(0, max(p99_e2e) * 1.2)
for bar, val in zip(bars4, p99_e2e):
    ax4.annotate(f'{val:.0f}', 
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add configuration details as text box
config_text = (
    "Configuration:\n"
    "• Aggregated: GH200 single node (cg1n1)\n"
    "• Intra-Node PD: 2 containers on GH200 (Mooncake)\n"
    "• Inter-Node PD: GH200 prefill → A100 decode (NIXL)"
)
fig.text(0.5, 0.02, config_text, ha='center', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig('/home/ubuntu/sglang-benchmark/benchmarks/results/max_config_comparison.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("Plot saved to: /home/ubuntu/sglang-benchmark/benchmarks/results/max_config_comparison.png")

# Also create a summary table
print("\n" + "="*80)
print("BENCHMARK RESULTS: Max Configuration (input=512, output=128, concurrency=32, n=100)")
print("="*80)
print(f"{'Configuration':<25} {'Throughput':>15} {'Mean TTFT':>12} {'Mean E2E':>12} {'P99 E2E':>12}")
print(f"{'':25} {'(tok/s)':>15} {'(ms)':>12} {'(ms)':>12} {'(ms)':>12}")
print("-"*80)
for i, cfg in enumerate(short_configs):
    print(f"{cfg:<25} {throughput[i]:>15.2f} {mean_ttft[i]:>12.2f} {mean_e2e[i]:>12.2f} {p99_e2e[i]:>12.2f}")
print("="*80)

