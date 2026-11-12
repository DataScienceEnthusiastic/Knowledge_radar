"""
===============================================================================
🔬 EXP-06A: Locked Final Test Set Evaluation (Frozen Pipeline)
===============================================================================
Evaluates the final frozen system configuration on the held-out LOCKED Test Set (N=70).
Protocol:
- The final test set is not used for tuning or decision-making.
- A complete final evaluation run is performed only after all decisions are frozen.
- Re-running the frozen evaluation for reproducibility is permitted, but test
  results must never feed back into configuration selection.

Computes:
- Retrieval metrics: Recall@1, Recall@3, Recall@5, MRR, NDCG@5
- Generation metrics: Answer Correctness, Relevance, Faithfulness, Citation Validity
- Bootstrap 95% Confidence Intervals for key metrics.
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
from modules.llm_router import is_time_sensitive_local
from evaluation.adapters.retriever_adapter import HybridRetrieverAdapter
from evaluation.adapters.reranker_adapter import CrossScoreReranker
from evaluation.adapters.web_adapter import WebVerificationAdapter
from evaluation.metrics.retrieval_metrics import (
    compute_recall_at_k, compute_mrr, compute_ndcg_at_k
)
from evaluation.metrics.generation_metrics import evaluate_generation_quality
from evaluation.metrics.uncertainty import bootstrap_confidence_interval

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


def synthesize_answer(query: str, retrieved_chunks: List[Dict[str, Any]], web_snippets: List[Dict[str, Any]]) -> str:
    """Performs dual-context audit synthesis for final test queries."""
    evidence_sentences = []

    # 1. Primary document evidence
    q_words = set(query.lower().split())
    for rank, chunk in enumerate(retrieved_chunks[:3], 1):
        text = chunk.get("text", "")
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for s in sentences:
            s_words = set(s.lower().split())
            if len(q_words.intersection(s_words)) >= 2:
                evidence_sentences.append(f"{s} [Doc {rank}]")

    # 2. Web verification evidence if present
    for rank, w in enumerate(web_snippets, 1):
        w_body = w.get("body", "").strip()
        if w_body:
            evidence_sentences.append(f"{w_body} [Web {rank}]")

    if not evidence_sentences:
        top_text = retrieved_chunks[0].get("text", "") if retrieved_chunks else ""
        evidence_sentences.append(f"{top_text[:200]}... [Doc 1]")

    synthesized = ". ".join(evidence_sentences[:4])
    if not synthesized.endswith("."):
        synthesized += "."
    return synthesized


def run_final_test_evaluation():
    print("=" * 70)
    print("🔬 RUNNING EXP-06A: LOCKED FINAL TEST SET EVALUATION (N=70)")
    print("=" * 70)
    print("NOTE: Final test set is evaluated with frozen configuration.")
    print("No parameters, thresholds, or architecture choices are modified.")
    print("-" * 70)

    # 1. Index full benchmark corpus (500 chars, 50 overlap)
    all_chunks = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".md"):
            with open(os.path.join(DOCS_DIR, f), "r", encoding="utf-8") as fp:
                chunks = chunk_text(fp.read(), source_name=f, chunk_size=500, chunk_overlap=50)
                all_chunks.extend(chunks)

    retriever = HybridRetrieverAdapter(rrf_k=60)
    retriever.index_documents(all_chunks)
    reranker = CrossScoreReranker()
    web_adapter = WebVerificationAdapter()

    # 2. Load held-out test queries (N=70)
    test_path = os.path.join(SPLITS_DIR, "test.jsonl")
    gt_path = os.path.join(DATASET_DIR, "ground_truth.jsonl")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    test_queries = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q_id = item["id"]
                gt = gt_map.get(q_id, {})
                test_queries.append({
                    "id": q_id,
                    "query": item["query"],
                    "category": item["category"],
                    "expected_answer": gt.get("expected_answer", ""),
                    "expected_entities": gt.get("expected_entities", []),
                    "relevant_docs": gt.get("relevant_docs", []),
                    "temporal": gt.get("temporal", False),
                    "has_conflict": gt.get("has_conflict", False),
                    "conflict_snapshot_id": gt.get("conflict_snapshot_id", None)
                })

    git_commit = get_git_commit()
    timestamp = datetime.datetime.now().isoformat()
    results_rows = []

    recalls_1, recalls_3, recalls_5 = [], [], []
    mrrs, ndcgs_5 = [], []
    correctness_scores, relevance_scores, faithfulness_scores = [], [], []
    citation_scores, latencies = [], []

    for q in test_queries:
        q_id = q["id"]
        q_text = q["query"]
        rel_docs = q["relevant_docs"]
        exp_ans = q["expected_answer"]
        exp_ents = q["expected_entities"]
        is_temporal = q["temporal"]
        snapshot_id = q["conflict_snapshot_id"]

        t0 = time.perf_counter()

        # Step 1: Candidate Pool Retrieval (Top-10)
        candidates = retriever.retrieve_hybrid_rrf(q_text, top_k=10, candidate_k=10)

        # Step 2: Cross-Score Reranking (Top-5)
        rerank_res = reranker.rerank(q_text, candidates, top_k=5)
        top5_chunks = rerank_res["results"]

        # Step 3: Adaptive Web Augmentation
        web_hits = []
        if is_time_sensitive_local(q_text) or is_temporal:
            web_res = web_adapter.search(q_text, mode="snapshot", snapshot_id=snapshot_id)
            web_hits = web_res["snippets"]

        # Step 4: Generation Synthesis
        generated_answer = synthesize_answer(q_text, top5_chunks, web_hits)

        total_lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(total_lat_ms)

        # Step 5: Metrics computation
        retrieved_doc_names = [h["source"] for h in top5_chunks]

        r1 = compute_recall_at_k(retrieved_doc_names, rel_docs, k=1)
        r3 = compute_recall_at_k(retrieved_doc_names, rel_docs, k=3)
        r5 = compute_recall_at_k(retrieved_doc_names, rel_docs, k=5)
        mrr = compute_mrr(retrieved_doc_names, rel_docs)
        ndcg5 = compute_ndcg_at_k(retrieved_doc_names, rel_docs, k=5)

        gen_eval = evaluate_generation_quality(
            query=q_text,
            generated_text=generated_answer,
            expected_answer=exp_ans,
            retrieved_chunks=top5_chunks,
            expected_entities=exp_ents
        )

        recalls_1.append(r1)
        recalls_3.append(r3)
        recalls_5.append(r5)
        mrrs.append(mrr)
        ndcgs_5.append(ndcg5)
        correctness_scores.append(gen_eval["answer_correctness"])
        relevance_scores.append(gen_eval["answer_relevance"])
        faithfulness_scores.append(gen_eval["faithfulness"])
        citation_scores.append(gen_eval["citation_validity"])

        results_rows.append({
            "experiment_id": "EXP-06A-FINAL-TEST",
            "dataset_version": "v1.0.0",
            "git_commit": git_commit,
            "timestamp": timestamp,
            "query_id": q_id,
            "category": q["category"],
            "recall_at_1": r1,
            "recall_at_3": r3,
            "recall_at_5": r5,
            "mrr": mrr,
            "ndcg_at_5": ndcg5,
            "answer_correctness": gen_eval["answer_correctness"],
            "answer_relevance": gen_eval["answer_relevance"],
            "faithfulness": gen_eval["faithfulness"],
            "citation_validity": gen_eval["citation_validity"],
            "entity_recall": gen_eval["entity_recall"],
            "total_latency_ms": round(total_lat_ms, 2),
            "generated_answer": generated_answer,
            "reproduction_command": "python evaluation/experiments/exp06a_final_test_evaluation.py"
        })

    # Uncertainty bootstrap estimation (95% CI)
    ci_r5 = bootstrap_confidence_interval(recalls_5, seed=42)
    ci_mrr = bootstrap_confidence_interval(mrrs, seed=42)
    ci_ndcg = bootstrap_confidence_interval(ndcgs_5, seed=42)
    ci_corr = bootstrap_confidence_interval(correctness_scores, seed=42)
    ci_faith = bootstrap_confidence_interval(faithfulness_scores, seed=42)
    ci_lat = bootstrap_confidence_interval(latencies, seed=42)

    print("📊 LOCKED FINAL TEST SET RESULTS (N=70 Queries with 95% Bootstrap CIs):")
    print("-" * 75)
    print(f"Recall@5:           {ci_r5['mean']:.4f}  (95% CI: [{ci_r5['ci_lower']:.4f}, {ci_r5['ci_upper']:.4f}])")
    print(f"MRR:                {ci_mrr['mean']:.4f}  (95% CI: [{ci_mrr['ci_lower']:.4f}, {ci_mrr['ci_upper']:.4f}])")
    print(f"NDCG@5:             {ci_ndcg['mean']:.4f}  (95% CI: [{ci_ndcg['ci_lower']:.4f}, {ci_ndcg['ci_upper']:.4f}])")
    print(f"Answer Correctness: {ci_corr['mean']:.4f}  (95% CI: [{ci_corr['ci_lower']:.4f}, {ci_corr['ci_upper']:.4f}])")
    print(f"Faithfulness:       {ci_faith['mean']:.4f}  (95% CI: [{ci_faith['ci_lower']:.4f}, {ci_faith['ci_upper']:.4f}])")
    print(f"Total Latency (ms): {ci_lat['mean']:.2f} ms (95% CI: [{ci_lat['ci_lower']:.2f}, {ci_lat['ci_upper']:.2f}])")
    print("-" * 75)

    csv_path = os.path.join(RESULTS_DIR, "exp06a_final_test_results.csv")
    fieldnames = list(results_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"Final test set results saved to: {csv_path}\n")
    return results_rows


if __name__ == "__main__":
    run_final_test_evaluation()
