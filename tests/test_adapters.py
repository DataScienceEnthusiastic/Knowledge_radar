"""
Unit tests verifying non-intrusive adapters (BM25, Dense, Hybrid RRF, Reranker, Web Adapter).
"""

import pytest
from evaluation.adapters import (
    BM25Retriever,
    HybridRetrieverAdapter,
    CrossScoreReranker,
    WebVerificationAdapter,
)


def test_bm25_retriever():
    retriever = BM25Retriever(k1=1.2, b=0.75)
    sample_chunks = [
        {"text": "Knowledge Radar uses 256-D vector embeddings for retrieval.", "source": "doc1", "chunk_index": 1},
        {"text": "Raft consensus provides leader election with randomized timeouts.", "source": "doc2", "chunk_index": 1},
        {"text": "PostgreSQL MVCC cleans dead tuples with autovacuum.", "source": "doc3", "chunk_index": 1}
    ]
    retriever.index_chunks(sample_chunks)

    results = retriever.search("Raft leader election", top_k=2)
    assert len(results) > 0
    assert results[0]["source"] == "doc2"
    assert results[0]["bm25_score"] > 0.0


def test_hybrid_rrf_retriever():
    adapter = HybridRetrieverAdapter(rrf_k=60)
    sample_chunks = [
        {"text": "Superconducting transmon qubit coherence T1 was 142 microseconds.", "source": "doc_quantum", "chunk_index": 1},
        {"text": "TechCorp total revenue in Q4 2024 was 4.82 Billion dollars.", "source": "doc_finance", "chunk_index": 1},
        {"text": "NVLink 4 provides 900 GB/s bandwidth per H100 GPU.", "source": "doc_nvlink", "chunk_index": 1}
    ]
    adapter.index_documents(sample_chunks)

    # Dense search
    dense_res = adapter.retrieve_dense("NVLink bandwidth", top_k=1)
    assert len(dense_res) == 1
    assert dense_res[0]["source"] == "doc_nvlink"

    # BM25 search
    bm25_res = adapter.retrieve_bm25("quantum qubit coherence", top_k=1)
    assert len(bm25_res) == 1
    assert bm25_res[0]["source"] == "doc_quantum"

    # Hybrid RRF
    hybrid_res = adapter.retrieve_hybrid_rrf("revenue TechCorp Q4 2024", top_k=1)
    assert len(hybrid_res) == 1
    assert hybrid_res[0]["source"] == "doc_finance"
    assert "rrf_score" in hybrid_res[0]


def test_cross_score_reranker():
    reranker = CrossScoreReranker()
    candidates = [
        {"text": "General cloud infrastructure notes and server management.", "source": "doc_a", "rank": 1},
        {"text": "FlashAttention-3 achieves 1.2 PFLOPs in FP8 on H100 GPUs using TMA.", "source": "doc_b", "rank": 2}
    ]
    query = "FlashAttention-3 FP8 throughput on H100"
    output = reranker.rerank(query, candidates, top_k=1)

    assert len(output["results"]) == 1
    assert output["results"][0]["source"] == "doc_b"
    assert output["results"][0]["rank"] == 1
    assert output["rerank_latency_ms"] >= 0.0


def test_web_verification_adapter_snapshot():
    web_adapter = WebVerificationAdapter()
    res = web_adapter.search("Knowledge Radar Pro pricing 2025", mode="snapshot", snapshot_id="kr_current_pricing_2025")

    assert len(res["snippets"]) > 0
    assert "35" in res["snippets"][0]["body"]
    assert res["mode"] == "snapshot"
