"""
EXP-01B: Comparative Chunking Strategy Benchmark.

Compares 3 distinct chunking paradigms across identical documents:
1. Fixed Sliding Window (500 chars, 50 chars overlap)
2. Sentence-Aware Splitting (max 500 chars, 1 overlap sentence)
3. Structure-Aware Heading Hierarchy (Title/section breadcrumb prefix + paragraph)

Evaluated on the Validation Set (N=50) across:
- Recall@1, Recall@3, Recall@5
- MRR (Mean Reciprocal Rank)
- NDCG@5 (deduplicated document hits)
- Total chunks generated
- Indexing latency and query retrieval latency
- Average top-5 context payload (characters)
"""

import os
import sys
import json
import time
import csv
import datetime
import subprocess
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.parser import chunk_text, chunk_sentences
from evaluation.adapters.retriever_adapter import HybridRetrieverAdapter
from evaluation.metrics.retrieval_metrics import (
    compute_recall_at_k, compute_mrr, compute_ndcg_at_k
)

DATASET_DIR = os.path.join(BASE_DIR, "evaluation", "dataset")
DOCS_DIR = os.path.join(DATASET_DIR, "documents")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_corpus_documents() -> Dict[str, str]:
    docs = {}
    for f in sorted(os.listdir(DOCS_DIR)):
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


def chunk_structure_aware(doc_name: str, text: str, max_chars: int = 500) -> List[Dict[str, Any]]:
    """
    Structure-aware chunking: extracts top-level markdown heading as a persistent
    section breadcrumb, attaching it to paragraphs to maintain context.
    """
    lines = text.strip().splitlines()
    title = ""
    body_lines = []
    for line in lines:
        if line.startswith("# ") and not title:
            title = line.lstrip("# ").strip()
        else:
            body_lines.append(line)

    body_text = "\n".join(body_lines).strip()
    prefix = f"[{title}]\n" if title else ""
    paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]

    chunks = []
    idx = 1
    for p in paragraphs:
        full_text = f"{prefix}{p}".strip()
        if len(full_text) <= max_chars:
            chunks.append({"text": full_text, "source": doc_name, "chunk_index": idx})
            idx += 1
        else:
            # Sub-split into sentence-aware blocks
            sub_chunks = chunk_sentences(full_text, source_name=doc_name, max_chunk_chars=max_chars)
            for sc in sub_chunks:
                sc["chunk_index"] = idx
                chunks.append(sc)
                idx += 1

    return chunks if chunks else chunk_text(text, source_name=doc_name, chunk_size=max_chars)


def run_strategy_benchmark():
    print("=" * 80)
    print("🔬 RUNNING EXP-01B: CHUNKING STRATEGY BENCHMARK (VALIDATION SET N=50)")
    print("=" * 80)

    docs = load_corpus_documents()
    queries = load_validation_queries()

    strategies = [
        ("Fixed Sliding Window (500/50)", lambda d, t: chunk_text(t, source_name=d, chunk_size=500, chunk_overlap=50)),
        ("Sentence-Aware Splitting (500)", lambda d, t: chunk_sentences(t, source_name=d, max_chunk_chars=500, overlap_sentences=1)),
        ("Structure-Aware Hierarchy (500)", lambda d, t: chunk_structure_aware(d, t, max_chars=500)),
    ]

    results_rows = []
    git_commit = get_git_commit()
    timestamp = datetime.datetime.now().isoformat()

    print(f"{'Strategy':<35}{'Chunks':<8}{'Recall@1':<10}{'Recall@3':<10}{'Recall@5':<10}{'MRR':<8}{'NDCG@5':<8}{'Lat(ms)':<8}{'Top-5 Chars':<12}")
    print("-" * 115)

    for strat_name, chunk_fn in strategies:
        # 1. Chunk and index corpus
        t0_idx = time.perf_counter()
        all_chunks = []
        for doc_name, text in docs.items():
            all_chunks.extend(chunk_fn(doc_name, text))
        idx_ms = (time.perf_counter() - t0_idx) * 1000.0

        adapter = HybridRetrieverAdapter(rrf_k=60)
        adapter.index_documents(all_chunks)

        # 2. Evaluate queries using Hybrid RRF
        recalls_1, recalls_3, recalls_5, mrrs, ndcgs_5, latencies, context_chars = [], [], [], [], [], [], []

        for q in queries:
            q_text = q["query"]
            rel_docs = q["relevant_docs"]

            t0_q = time.perf_counter()
            hits = adapter.retrieve_hybrid_rrf(q_text, top_k=5, candidate_k=20)
            lat_ms = (time.perf_counter() - t0_q) * 1000.0
            latencies.append(lat_ms)
            context_chars.append(sum(len(h.get("text", "")) for h in hits))

            retrieved_doc_names = [h["source"] for h in hits]
            recalls_1.append(compute_recall_at_k(retrieved_doc_names, rel_docs, k=1))
            recalls_3.append(compute_recall_at_k(retrieved_doc_names, rel_docs, k=3))
            recalls_5.append(compute_recall_at_k(retrieved_doc_names, rel_docs, k=5))
            mrrs.append(compute_mrr(retrieved_doc_names, rel_docs))
            ndcgs_5.append(compute_ndcg_at_k(retrieved_doc_names, rel_docs, k=5))

        m_r1 = round(sum(recalls_1) / len(recalls_1), 4)
        m_r3 = round(sum(recalls_3) / len(recalls_3), 4)
        m_r5 = round(sum(recalls_5) / len(recalls_5), 4)
        m_mrr = round(sum(mrrs) / len(mrrs), 4)
        m_ndcg = round(sum(ndcgs_5) / len(ndcgs_5), 4)
        m_lat = round(sum(latencies) / len(latencies), 3)
        m_ctx = round(sum(context_chars) / len(context_chars), 1)

        print(f"{strat_name:<35}{len(all_chunks):<8}{m_r1:<10}{m_r3:<10}{m_r5:<10}{m_mrr:<8}{m_ndcg:<8}{m_lat:<8}{m_ctx:<12}")

        results_rows.append({
            "experiment_id": "EXP-01B-CHUNKING-STRATEGIES",
            "dataset_version": "v1.0.0",
            "git_commit": git_commit,
            "timestamp": timestamp,
            "strategy": strat_name,
            "total_chunks": len(all_chunks),
            "indexing_time_ms": round(idx_ms, 2),
            "recall_at_1": m_r1,
            "recall_at_3": m_r3,
            "recall_at_5": m_r5,
            "mrr": m_mrr,
            "ndcg_at_5": m_ndcg,
            "avg_retrieval_latency_ms": m_lat,
            "avg_context_chars_top5": m_ctx,
            "reproduction_command": "python evaluation/experiments/exp01b_chunking_strategies.py"
        })

    csv_path = os.path.join(RESULTS_DIR, "exp01b_chunking_strategies.csv")
    fieldnames = list(results_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print("-" * 115)
    print(f"Results saved to: {csv_path}\n")
    return results_rows


run_chunking_strategies_experiment = run_strategy_benchmark

if __name__ == "__main__":
    run_strategy_benchmark()
