"""
===============================================================================
🔬 EXP-05: Generation Quality, Faithfulness & Groundedness Audit
===============================================================================
Evaluates the synthesis stage independently from retrieval quality on the
Validation Set (N=50).
Measures:
- Answer Correctness (Semantic + Entity agreement with ground truth)
- Answer Relevance (Query alignment without evasion)
- Faithfulness / Groundedness (Claim support by retrieved context)
- Evidence & Citation Support (Citation ID validity)
- Unsupported Claim Rate (Hallucinations per 100 tokens)
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
from evaluation.metrics.generation_metrics import evaluate_generation_quality

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


def synthesize_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Simulates / performs context synthesis using retrieved evidence chunks.
    Constructs a grounded, cited response following Knowledge Radar's synthesis format.
    """
    if not retrieved_chunks:
        return "I do not have sufficient information in the provided context to answer this query."

    # Extract sentences containing key matching terms from top chunks
    q_words = set(query.lower().split())
    evidence_sentences = []

    for rank, chunk in enumerate(retrieved_chunks[:3], 1):
        text = chunk.get("text", "")
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for s in sentences:
            s_words = set(s.lower().split())
            if len(q_words.intersection(s_words)) >= 2:
                evidence_sentences.append(f"{s} [Doc {rank}]")

    if not evidence_sentences:
        # Fallback to the top chunk's first sentence
        top_text = retrieved_chunks[0].get("text", "")
        evidence_sentences.append(f"{top_text[:200]}... [Doc 1]")

    # Format synthesized response
    synthesized = ". ".join(evidence_sentences[:3])
    if not synthesized.endswith("."):
        synthesized += "."
    return synthesized


def run_generation_quality_experiment():
    print("=" * 70)
    print("🔬 RUNNING EXP-05: GENERATION QUALITY & GROUNDEDNESS AUDIT (VALIDATION SET N=50)")
    print("=" * 70)

    # 1. Index corpus
    all_chunks = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".md"):
            with open(os.path.join(DOCS_DIR, f), "r", encoding="utf-8") as fp:
                chunks = chunk_text(fp.read(), source_name=f, chunk_size=500, chunk_overlap=50)
                all_chunks.extend(chunks)

    retriever = HybridRetrieverAdapter(rrf_k=60)
    retriever.index_documents(all_chunks)
    reranker = CrossScoreReranker()

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
                gt = gt_map.get(q_id, {})
                val_queries.append({
                    "id": q_id,
                    "query": item["query"],
                    "category": item["category"],
                    "expected_answer": gt.get("expected_answer", ""),
                    "expected_entities": gt.get("expected_entities", []),
                    "relevant_docs": gt.get("relevant_docs", [])
                })

    git_commit = get_git_commit()
    timestamp = datetime.datetime.now().isoformat()
    results_rows = []

    correctness_scores = []
    relevance_scores = []
    faithfulness_scores = []
    unsupported_claim_rates = []
    citation_validities = []

    print(f"{'Query ID':<10}{'Category':<16}{'Correctness':<14}{'Relevance':<12}{'Faithfulness':<14}{'Unsupp. Rate':<14}{'Citations':<10}")
    print("-" * 90)

    for q in val_queries:
        q_id = q["id"]
        q_text = q["query"]
        exp_ans = q["expected_answer"]
        exp_ents = q["expected_entities"]

        # Step 1: Retrieval + Reranking to obtain Top-5
        candidates = retriever.retrieve_hybrid_rrf(q_text, top_k=10, candidate_k=10)
        rerank_res = reranker.rerank(q_text, candidates, top_k=5)
        top5_chunks = rerank_res["results"]

        # Step 2: Synthesis
        generated_answer = synthesize_answer(q_text, top5_chunks)

        # Step 3: Evaluation
        metrics = evaluate_generation_quality(
            query=q_text,
            generated_text=generated_answer,
            expected_answer=exp_ans,
            retrieved_chunks=top5_chunks,
            expected_entities=exp_ents
        )

        correctness_scores.append(metrics["answer_correctness"])
        relevance_scores.append(metrics["answer_relevance"])
        faithfulness_scores.append(metrics["faithfulness"])
        unsupported_claim_rates.append(metrics["unsupported_claim_rate"])
        citation_validities.append(metrics["citation_validity"])

        results_rows.append({
            "experiment_id": "EXP-05-GENERATION-QUALITY",
            "dataset_version": "v1.0.0",
            "git_commit": git_commit,
            "timestamp": timestamp,
            "query_id": q_id,
            "category": q["category"],
            "answer_correctness": metrics["answer_correctness"],
            "answer_relevance": metrics["answer_relevance"],
            "faithfulness": metrics["faithfulness"],
            "unsupported_claim_rate": metrics["unsupported_claim_rate"],
            "citation_validity": metrics["citation_validity"],
            "entity_recall": metrics["entity_recall"],
            "context_has_evidence": metrics["context_has_evidence"],
            "reproduction_command": "python evaluation/experiments/exp05_generation_quality.py"
        })

    mean_corr = round(sum(correctness_scores) / len(correctness_scores), 4)
    mean_rel = round(sum(relevance_scores) / len(relevance_scores), 4)
    mean_faith = round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
    mean_unsupp = round(sum(unsupported_claim_rates) / len(unsupported_claim_rates), 2)
    mean_cite = round(sum(citation_validities) / len(citation_validities), 4)

    print("-" * 90)
    print(f"{'MEAN (N=50)':<26}{mean_corr:<14}{mean_rel:<12}{mean_faith:<14}{mean_unsupp:<14}{mean_cite:<10}")

    csv_path = os.path.join(RESULTS_DIR, "exp05_generation_quality.csv")
    fieldnames = list(results_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\nResults successfully saved to: {csv_path}\n")
    return results_rows


if __name__ == "__main__":
    run_generation_quality_experiment()
