"""
===============================================================================
🔬 EXP-02: Dense vs. Lexical vs. Hybrid Retrieval (No Reranking)
===============================================================================
Compares Dense Vector retrieval, Pure Okapi BM25, and Reciprocal Rank Fusion
(RRF k=60) across 7 query categories on the Validation Set (N=50).
Strictly does NOT include reranking to isolate core retrieval behavior.
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


def run_retrieval_baselines_experiment():
    print("=" * 70)
    print("🔬 RUNNING EXP-02: DENSE vs. BM25 vs. HYBRID RRF (VALIDATION SET N=50)")
    print("=" * 70)

    # 1. Index corpus using standard default chunking (500 chars, 50 overlap)
    all_chunks = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".md"):
            with open(os.path.join(DOCS_DIR, f), "r", encoding="utf-8") as fp:
                chunks = chunk_text(fp.read(), source_name=f, chunk_size=500, chunk_overlap=50)
                all_chunks.extend(chunks)

    adapter = HybridRetrieverAdapter(rrf_k=60)
    adapter.index_documents(all_chunks)

    # 2. Load validation queries and ground truth
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
                    "category": item["category"],
                    "relevant_docs": gt_map.get(q_id, {}).get("relevant_docs", [])
                })

    categories = sorted(list(set(q["category"] for q in val_queries)))
    strategies = ["Dense (Cosine)", "BM25 (Okapi)", "Hybrid (RRF k=60)"]

    git_commit = get_git_commit()
    timestamp = datetime.datetime.now().isoformat()
    results_rows = []

    print(f"{'Strategy':<20}{'Category':<16}{'Count':<8}{'Recall@1':<10}{'Recall@3':<10}{'Recall@5':<10}{'MRR':<8}{'NDCG@5':<8}{'Lat(ms)':<8}")
    print("-" * 100)

    for strat in strategies:
        # Run over each category
        for cat in categories + ["ALL"]:
            if cat == "ALL":
                cat_queries = val_queries
            else:
                cat_queries = [q for q in val_queries if q["category"] == cat]

            if not cat_queries:
                continue

            recalls_1 = []
            recalls_3 = []
            recalls_5 = []
            mrrs = []
            ndcgs_5 = []
            latencies = []
            context_chars = []

            for q in cat_queries:
                q_text = q["query"]
                rel_docs = q["relevant_docs"]

                t0 = time.perf_counter()
                if strat == "Dense (Cosine)":
                    hits = adapter.retrieve_dense(q_text, top_k=5)
                elif strat == "BM25 (Okapi)":
                    hits = adapter.retrieve_bm25(q_text, top_k=5)
                elif strat == "Hybrid (RRF k=60)":
                    hits = adapter.retrieve_hybrid_rrf(q_text, top_k=5, candidate_k=20)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat_ms)
                context_chars.append(sum(len(h.get("text", "")) for h in hits))

                retrieved_doc_names = [h["source"] for h in hits]

                recalls_1.append(compute_recall_at_k(retrieved_doc_names, rel_docs, k=1))
                recalls_3.append(compute_recall_at_k(retrieved_doc_names, rel_docs, k=3))
                recalls_5.append(compute_recall_at_k(retrieved_doc_names, rel_docs, k=5))
                mrrs.append(compute_mrr(retrieved_doc_names, rel_docs))
                ndcgs_5.append(compute_ndcg_at_k(retrieved_doc_names, rel_docs, k=5))

            mean_r1 = round(sum(recalls_1) / len(recalls_1), 4)
            mean_r3 = round(sum(recalls_3) / len(recalls_3), 4)
            mean_r5 = round(sum(recalls_5) / len(recalls_5), 4)
            mean_mrr = round(sum(mrrs) / len(mrrs), 4)
            mean_ndcg5 = round(sum(ndcgs_5) / len(ndcgs_5), 4)
            mean_lat = round(sum(latencies) / len(latencies), 3)
            mean_ctx = round(sum(context_chars) / len(context_chars), 1)

            print(f"{strat:<20}{cat:<16}{len(cat_queries):<8}{mean_r1:<10}{mean_r3:<10}{mean_r5:<10}{mean_mrr:<8}{mean_ndcg5:<8}{mean_lat:<8}")

            results_rows.append({
                "experiment_id": "EXP-02-RETRIEVAL-BASELINES",
                "dataset_version": "v1.0.0",
                "git_commit": git_commit,
                "timestamp": timestamp,
                "strategy": strat,
                "category": cat,
                "sample_count": len(cat_queries),
                "recall_at_1": mean_r1,
                "recall_at_3": mean_r3,
                "recall_at_5": mean_r5,
                "mrr": mean_mrr,
                "ndcg_at_5": mean_ndcg5,
                "avg_retrieval_latency_ms": mean_lat,
                "avg_context_chars_top5": mean_ctx,
                "reproduction_command": "python evaluation/experiments/exp02_retrieval_baselines.py"
            })

    csv_path = os.path.join(RESULTS_DIR, "exp02_retrieval_by_category.csv")
    fieldnames = list(results_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print("-" * 100)
    print(f"Results successfully saved to: {csv_path}\n")
    return results_rows


if __name__ == "__main__":
    run_retrieval_baselines_experiment()
