"""
===============================================================================
🔬 EXP-01: Document Chunking Factorial Optimization
===============================================================================
Evaluates a 4x3 factorial matrix of chunk sizes and overlaps on the
Validation Set (N=50) to measure the tradeoff between retrieval recall,
granularity, context token footprint, and retrieval latency.

Factorial Grid:
- Chunk Sizes (chars): [256, 500, 750, 1000]
- Overlaps (%): [0%, 10%, 20%]
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


def load_corpus_documents() -> Dict[str, str]:
    docs = {}
    for f in os.listdir(DOCS_DIR):
        if f.endswith(".md"):
            with open(os.path.join(DOCS_DIR, f), "r", encoding="utf-8") as fp:
                docs[f] = fp.read()
    return docs


def load_validation_queries() -> List[Dict[str, Any]]:
    val_path = os.path.join(SPLITS_DIR, "validation.jsonl")
    gt_path = os.path.join(DATASET_DIR, "ground_truth.jsonl")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    queries = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q_id = item["id"]
                gt = gt_map.get(q_id, {})
                queries.append({
                    "id": q_id,
                    "query": item["query"],
                    "category": item["category"],
                    "relevant_docs": gt.get("relevant_docs", [])
                })
    return queries


def run_chunking_experiment():
    print("=" * 70)
    print("🔬 RUNNING EXP-01: CHUNKING FACTORIAL OPTIMIZATION (VALIDATION SET N=50)")
    print("=" * 70)

    corpus_docs = load_corpus_documents()
    val_queries = load_validation_queries()
    git_commit = get_git_commit()
    timestamp = datetime.datetime.now().isoformat()

    chunk_sizes = [256, 500, 750, 1000]
    overlap_ratios = [0.0, 0.10, 0.20]

    results_rows = []

    print(f"{'Chunk Size':<12}{'Overlap':<10}{'Total Chunks':<14}{'Recall@1':<10}{'Recall@3':<10}{'Recall@5':<10}{'MRR':<8}{'NDCG@5':<8}{'Lat(ms)':<8}")
    print("-" * 90)

    for c_size in chunk_sizes:
        for o_ratio in overlap_ratios:
            overlap_chars = int(c_size * o_ratio)
            overlap_pct_str = f"{int(o_ratio * 100)}%"

            # 1. Chunk full corpus
            t_chunk_start = time.perf_counter()
            all_chunks = []
            for doc_name, doc_text in corpus_docs.items():
                doc_chunks = chunk_text(doc_text, source_name=doc_name, chunk_size=c_size, chunk_overlap=overlap_chars)
                all_chunks.extend(doc_chunks)
            indexing_time_ms = (time.perf_counter() - t_chunk_start) * 1000.0

            # 2. Index in retriever
            adapter = HybridRetrieverAdapter(rrf_k=60)
            adapter.index_documents(all_chunks)

            # 3. Evaluate retrieval on validation set
            recalls_1 = []
            recalls_3 = []
            recalls_5 = []
            mrrs = []
            ndcgs_5 = []
            latencies = []
            context_chars = []

            for q in val_queries:
                query_text = q["query"]
                rel_docs = q["relevant_docs"]

                t0 = time.perf_counter()
                hits = adapter.retrieve_dense(query_text, top_k=5)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat_ms)

                retrieved_doc_names = [h["source"] for h in hits]
                total_chars = sum(len(h.get("text", "")) for h in hits)
                context_chars.append(total_chars)

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
            mean_chars = round(sum(context_chars) / len(context_chars), 1)

            print(f"{c_size:<12}{overlap_pct_str:<10}{len(all_chunks):<14}{mean_r1:<10}{mean_r3:<10}{mean_r5:<10}{mean_mrr:<8}{mean_ndcg5:<8}{mean_lat:<8}")

            results_rows.append({
                "experiment_id": "EXP-01-CHUNKING",
                "dataset_version": "v1.0.0",
                "git_commit": git_commit,
                "timestamp": timestamp,
                "chunk_size": c_size,
                "overlap_pct": overlap_pct_str,
                "overlap_chars": overlap_chars,
                "total_chunks_indexed": len(all_chunks),
                "indexing_time_ms": round(indexing_time_ms, 2),
                "recall_at_1": mean_r1,
                "recall_at_3": mean_r3,
                "recall_at_5": mean_r5,
                "mrr": mean_mrr,
                "ndcg_at_5": mean_ndcg5,
                "avg_retrieval_latency_ms": mean_lat,
                "avg_context_chars_top5": mean_chars,
                "reproduction_command": "python evaluation/experiments/exp01_chunking.py"
            })

    csv_path = os.path.join(RESULTS_DIR, "exp01_chunking_results.csv")
    fieldnames = list(results_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictDict = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print("-" * 90)
    print(f"Results successfully saved to: {csv_path}\n")
    return results_rows


if __name__ == "__main__":
    run_chunking_experiment()
