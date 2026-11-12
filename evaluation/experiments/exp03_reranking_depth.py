"""
===============================================================================
🔬 EXP-03: Reranking Depth & Computational Cost-Benefit Tradeoff
===============================================================================
Evaluates candidate pool reranking over the winning retrieval baseline (Hybrid RRF)
on the Validation Set (N=50) across three depth configurations:
1. Top-5 Retrieved Directly (No Reranker)
2. Top-10 Candidates -> Cross-Score Rerank -> Top-5
3. Top-20 Candidates -> Cross-Score Rerank -> Top-5

Measures:
- Quality: Recall@5, MRR, NDCG@5
- Cost/Performance: Retrieval Latency, Rerank Latency, Total Latency, Context Chars
===============================================================================
"""

import os
import sys
import json
import time
import csv
import datetime
import subprocess
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.parser import chunk_text
from evaluation.adapters.retriever_adapter import HybridRetrieverAdapter
from evaluation.adapters.reranker_adapter import CrossScoreReranker
from evaluation.metrics.retrieval_metrics import (
    compute_recall_at_k, compute_mrr, compute_ndcg_at_k
)

DATASET_DIR = os.path.join(BASE_DIR, "evaluation", "dataset")
DOCS_DIR = os.path.join(DATASET_DIR, "documents")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_reranking_depth_experiment():
    print("=" * 70)
    print("🔬 RUNNING EXP-03: RERANKING DEPTH & COST-BENEFIT TRADEOFF (VALIDATION SET N=50)")
    print("=" * 70)

    # 1. Index corpus
    all_chunks = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".md"):
            with open(os.path.join(DOCS_DIR, f), "r", encoding="utf-8") as fp:
                chunks = chunk_text(fp.read(), source_name=f, chunk_size=500, chunk_overlap=50)
                all_chunks.extend(chunks)

    adapter = HybridRetrieverAdapter(rrf_k=60)
    adapter.index_documents(all_chunks)
    reranker = CrossScoreReranker()

    # 2. Load validation queries
    val_path = os.path.join(SPLITS_DIR, "validation.jsonl")
    gt_path = os.path.join(DATASET_DIR, "ground_truth.jsonl")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    val_queries = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q_id = item["id"]
                val_queries.append({
                    "id": q_id,
                    "query": item["query"],
                    "relevant_docs": gt_map.get(q_id, {}).get("relevant_docs", [])
                })

    configs = [
        {"name": "Direct Top-5 (No Rerank)", "candidate_k": 5, "rerank": False},
        {"name": "Top-10 -> Rerank -> Top-5", "candidate_k": 10, "rerank": True},
        {"name": "Top-20 -> Rerank -> Top-5", "candidate_k": 20, "rerank": True}
    ]

    git_commit = get_git_commit()
    timestamp = datetime.datetime.now().isoformat()
    results_rows = []

    print(f"{'Configuration':<32}{'Recall@5':<12}{'MRR':<10}{'NDCG@5':<10}{'Retr Lat(ms)':<14}{'Rerank Lat(ms)':<16}{'Total Lat(ms)':<14}")
    print("-" * 105)

    for cfg in configs:
        recalls_5 = []
        mrrs = []
        ndcgs_5 = []
        retr_latencies = []
        rerank_latencies = []
        total_latencies = []
        context_chars = []

        for q in val_queries:
            q_text = q["query"]
            rel_docs = q["relevant_docs"]

            t0 = time.perf_counter()
            initial_hits = adapter.retrieve_hybrid_rrf(q_text, top_k=cfg["candidate_k"], candidate_k=cfg["candidate_k"])
            retr_ms = (time.perf_counter() - t0) * 1000.0

            if cfg["rerank"]:
                rerank_out = reranker.rerank(q_text, initial_hits, top_k=5)
                final_hits = rerank_out["results"]
                rerank_ms = rerank_out["rerank_latency_ms"]
            else:
                final_hits = initial_hits[:5]
                rerank_ms = 0.0

            total_ms = retr_ms + rerank_ms

            retr_latencies.append(retr_ms)
            rerank_latencies.append(rerank_ms)
            total_latencies.append(total_ms)

            retrieved_doc_names = [h["source"] for h in final_hits]
            total_chars = sum(len(h.get("text", "")) for h in final_hits)
            context_chars.append(total_chars)

            recalls_5.append(compute_recall_at_k(retrieved_doc_names, rel_docs, k=5))
            mrrs.append(compute_mrr(retrieved_doc_names, rel_docs))
            ndcgs_5.append(compute_ndcg_at_k(retrieved_doc_names, rel_docs, k=5))

        mean_r5 = round(sum(recalls_5) / len(recalls_5), 4)
        mean_mrr = round(sum(mrrs) / len(mrrs), 4)
        mean_ndcg5 = round(sum(ndcgs_5) / len(ndcgs_5), 4)
        mean_retr_lat = round(sum(retr_latencies) / len(retr_latencies), 3)
        mean_rerank_lat = round(sum(rerank_latencies) / len(rerank_latencies), 3)
        mean_total_lat = round(sum(total_latencies) / len(total_latencies), 3)
        mean_chars = round(sum(context_chars) / len(context_chars), 1)

        print(f"{cfg['name']:<32}{mean_r5:<12}{mean_mrr:<10}{mean_ndcg5:<10}{mean_retr_lat:<14}{mean_rerank_lat:<16}{mean_total_lat:<14}")

        results_rows.append({
            "experiment_id": "EXP-03-RERANKING-DEPTH",
            "dataset_version": "v1.0.0",
            "git_commit": git_commit,
            "timestamp": timestamp,
            "configuration": cfg["name"],
            "candidate_pool_k": cfg["candidate_k"],
            "final_context_k": 5,
            "recall_at_5": mean_r5,
            "mrr": mean_mrr,
            "ndcg_at_5": mean_ndcg5,
            "avg_retrieval_latency_ms": mean_retr_lat,
            "avg_reranking_latency_ms": mean_rerank_lat,
            "avg_total_latency_ms": mean_total_lat,
            "avg_context_chars_top5": mean_chars,
            "reproduction_command": "python evaluation/experiments/exp03_reranking_depth.py"
        })

    csv_path = os.path.join(RESULTS_DIR, "exp03_reranking_tradeoffs.csv")
    fieldnames = list(results_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print("-" * 105)
    print(f"Results successfully saved to: {csv_path}\n")
    return results_rows


if __name__ == "__main__":
    run_reranking_depth_experiment()
