"""
===============================================================================
📊 PLOT GENERATOR: Decision-Oriented Visualizations (Matplotlib Only)
===============================================================================
Generates clean, interpretable, decision-oriented charts from raw results CSVs:
1. plot_exp01_chunking_tradeoff.png (Recall@5 vs Chunks & Context)
2. plot_exp02_retrieval_by_category.png (Dense vs BM25 vs Hybrid across 7 categories)
3. plot_exp03_reranking_cost_frontier.png (Quality Gain vs Latency Overhead)
4. plot_exp04_web_verification_latency_noise.png (Temporal Accuracy vs Noise vs Latency)
5. plot_exp06_error_waterfall.png (Attrition Waterfall from Retrieval to Correctness)
6. plot_exp06_failure_distribution.png (10-Point Failure Mode Breakdown)
===============================================================================
"""

import os
import sys
import csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")
PLOTS_DIR = os.path.join(BASE_DIR, "evaluation", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#EEEEEE'
    plt.rcParams['grid.linestyle'] = '--'
except ImportError:
    HAS_MATPLOTLIB = False


def plot_exp01():
    csv_file = os.path.join(RESULTS_DIR, "exp01_chunking_results.csv")
    if not os.path.exists(csv_file):
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    # Group by chunk size for 10% overlap
    ten_pct = [d for d in data if d["overlap_pct"] == "10%"]
    if not ten_pct:
        ten_pct = data[::3]

    sizes = [int(d["chunk_size"]) for d in ten_pct]
    recall5 = [float(d["recall_at_5"]) for d in ten_pct]
    latency = [float(d["avg_retrieval_latency_ms"]) for d in ten_pct]
    mrr = [float(d["mrr"]) for d in ten_pct]

    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=150)

    color1 = '#1E88E5'
    color2 = '#E53935'

    ax1.set_xlabel('Chunk Size (Characters)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Recall@5 & MRR', color=color1, fontsize=11, fontweight='bold', labelpad=8)
    line1 = ax1.plot(sizes, recall5, marker='o', color=color1, linewidth=2, label='Recall@5')
    line2 = ax1.plot(sizes, mrr, marker='s', color='#004BA0', linewidth=2, linestyle='--', label='MRR')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Retrieval Latency (ms)', color=color2, fontsize=11, fontweight='bold', labelpad=8)
    line3 = ax2.plot(sizes, latency, marker='^', color=color2, linewidth=2, linestyle='-.', label='Latency (ms)')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', framealpha=0.9)

    plt.title('EXP-01: Chunk Size vs. Retrieval Recall & Latency (10% Overlap)', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "plot_exp01_chunking_tradeoff.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_exp02():
    csv_file = os.path.join(RESULTS_DIR, "exp02_retrieval_by_category.csv")
    if not os.path.exists(csv_file):
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    categories = ["factual", "keyword-heavy", "semantic", "multi-hop", "entity-specific", "temporal", "comparison"]
    dense_r5 = []
    bm25_r5 = []
    hybrid_r5 = []

    for cat in categories:
        d_val = next((float(d["recall_at_5"]) for d in data if d["category"] == cat and "Dense" in d["strategy"]), 0.0)
        b_val = next((float(d["recall_at_5"]) for d in data if d["category"] == cat and "BM25" in d["strategy"]), 0.0)
        h_val = next((float(d["recall_at_5"]) for d in data if d["category"] == cat and "Hybrid" in d["strategy"]), 0.0)
        dense_r5.append(d_val)
        bm25_r5.append(b_val)
        hybrid_r5.append(h_val)

    x = np.arange(len(categories))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    rects1 = ax.bar(x - width, dense_r5, width, label='Dense (Cosine)', color='#42A5F5', edgecolor='#1976D2')
    rects2 = ax.bar(x, bm25_r5, width, label='BM25 (Okapi)', color='#FFA726', edgecolor='#F57C00')
    rects3 = ax.bar(x + width, hybrid_r5, width, label='Hybrid (RRF k=60)', color='#66BB6A', edgecolor='#388E3C')

    ax.set_ylabel('Recall@5', fontsize=11, fontweight='bold')
    ax.set_title('EXP-02: Retrieval Recall@5 by Query Category (Validation Set N=50)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right', fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y')

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "plot_exp02_retrieval_by_category.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_exp03():
    csv_file = os.path.join(RESULTS_DIR, "exp03_reranking_tradeoffs.csv")
    if not os.path.exists(csv_file):
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    configs = [d["configuration"] for d in data]
    ndcg5 = [float(d["ndcg_at_5"]) for d in data]
    recall5 = [float(d["recall_at_5"]) for d in data]
    total_lat = [float(d["avg_total_latency_ms"]) for d in data]

    x = np.arange(len(configs))

    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=150)

    width = 0.35
    rects1 = ax1.bar(x - width/2, recall5, width, label='Recall@5', color='#26A69A', edgecolor='#00695C')
    rects2 = ax1.bar(x + width/2, ndcg5, width, label='NDCG@5', color='#5C6BC0', edgecolor='#283593')
    ax1.set_ylabel('Retrieval Quality Score', fontsize=11, fontweight='bold')
    ax1.set_ylim(0.0, 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Direct Top-5", "Top-10 -> Rerank", "Top-20 -> Rerank"], fontsize=10, fontweight='bold')
    ax1.grid(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(x, total_lat, color='#D32F2F', marker='D', linewidth=2.5, label='Total Latency (ms)')
    ax2.set_ylabel('End-to-End Latency (ms)', color='#D32F2F', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#D32F2F')

    ax1.legend(loc='lower left', framealpha=0.9)
    ax2.legend(loc='lower right', framealpha=0.9)

    plt.title('EXP-03: Reranking Depth — Quality Gain vs. Latency Overhead', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "plot_exp03_reranking_cost_frontier.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_exp04():
    csv_file = os.path.join(RESULTS_DIR, "exp04_web_verification_results.csv")
    if not os.path.exists(csv_file):
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    modes = ["Docs Only", "Always Web", "Adaptive Conditional"]
    t_acc = [float(d["temporal_accuracy"]) for d in data]
    c_f1 = [float(d["conflict_f1"]) for d in data]
    noise_tok = [float(d["avg_injected_noise_tokens_on_static"]) for d in data]

    x = np.arange(len(modes))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=150)

    rects1 = ax1.bar(x - width/2, t_acc, width, label='Temporal Accuracy', color='#42A5F5', edgecolor='#1565C0')
    rects2 = ax1.bar(x + width/2, c_f1, width, label='Conflict F1 Score', color='#AB47BC', edgecolor='#6A1B9A')
    ax1.set_ylabel('Audit & Freshness Score', fontsize=11, fontweight='bold')
    ax1.set_ylim(0.0, 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, fontsize=10, fontweight='bold')
    ax1.grid(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(x, noise_tok, color='#E65100', marker='o', linewidth=2.5, label='Static Noise (Tokens)')
    ax2.set_ylabel('Extraneous Noise Tokens Injected', color='#E65100', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#E65100')

    ax1.legend(loc='upper left', framealpha=0.9)
    ax2.legend(loc='upper center', framealpha=0.9)

    plt.title('EXP-04: Web Augmentation — Freshness Gain vs. Noise Penalty', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "plot_exp04_web_verification_latency_noise.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_exp06():
    # 1. Waterfall plot
    wf_file = os.path.join(RESULTS_DIR, "exp06b_error_waterfall.csv")
    if os.path.exists(wf_file):
        with open(wf_file, "r", encoding="utf-8") as f:
            wf_data = list(csv.DictReader(f))

        stages = ["Total Test Queries\n(N=70)", "Retrieval Success\n(Recall@5 > 0)", "Citation Support\n(Citations Valid)", "Faithfulness\n(>= 0.70)", "Answer Correctness\n(>= 0.70)"]
        counts = [int(d["count"]) for d in wf_data]
        pcts = [float(d["pct_of_total"]) for d in wf_data]

        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
        colors = ['#1E88E5', '#26A69A', '#7E57C2', '#FFA726', '#66BB6A']
        bars = ax.bar(stages, counts, color=colors, edgecolor='#333333', width=0.55)

        for bar, count, pct in zip(bars, counts, pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1.2, f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

        ax.set_ylabel('Query Count', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 85)
        ax.set_title('EXP-06B: Error Waterfall Cascade on Held-Out Test Set (N=70)', fontsize=12, fontweight='bold', pad=12)
        ax.grid(axis='y')

        plt.tight_layout()
        wf_plot_path = os.path.join(PLOTS_DIR, "plot_exp06_error_waterfall.png")
        plt.savefig(wf_plot_path)
        plt.close()
        print(f"Saved: {wf_plot_path}")

    # 2. Failure taxonomy bar chart
    tax_file = os.path.join(RESULTS_DIR, "exp06b_failure_taxonomy.csv")
    if os.path.exists(tax_file):
        with open(tax_file, "r", encoding="utf-8") as f:
            tax_data = list(csv.DictReader(f))

        # Filter categories with counts > 0
        active_cats = [d for d in tax_data if int(d["failure_count"]) > 0]
        if not active_cats:
            active_cats = tax_data[:5]

        cat_names = [d["failure_category"].replace("_", " ") for d in active_cats]
        cat_counts = [int(d["failure_count"]) for d in active_cats]

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        y_pos = np.arange(len(cat_names))

        ax.barh(y_pos, cat_counts, color='#EF5350', edgecolor='#B71C1C', height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cat_names, fontsize=9.5, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('Failure Frequency Count (Test Set N=70)', fontsize=11, fontweight='bold')
        ax.set_title('EXP-06B: 10-Point Failure Taxonomy Distribution', fontsize=12, fontweight='bold', pad=12)
        ax.grid(axis='x')

        for i, v in enumerate(cat_counts):
            ax.text(v + 0.15, i, str(v), va='center', fontsize=9.5, fontweight='bold')

        plt.tight_layout()
        tax_plot_path = os.path.join(PLOTS_DIR, "plot_exp06_failure_distribution.png")
        plt.savefig(tax_plot_path)
        plt.close()
        print(f"Saved: {tax_plot_path}")


def generate_all_plots():
    if not HAS_MATPLOTLIB:
        print("Notice: matplotlib is not installed in the active environment. Skipping plot generation.")
        return
    print("=" * 70)
    print("📊 GENERATING DECISION-ORIENTED PLOTS (MATPLOTLIB ONLY)")
    print("=" * 70)
    plot_exp01()
    plot_exp02()
    plot_exp03()
    plot_exp04()
    plot_exp06()
    print("All plots generated successfully in evaluation/plots/\n")


if __name__ == "__main__":
    generate_all_plots()
