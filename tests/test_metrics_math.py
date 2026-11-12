"""
Unit tests for evaluation metrics mathematics (IR metrics, uncertainty CI, audit metrics).
"""

import pytest
import math
from evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    bootstrap_confidence_interval,
    compute_conflict_detection_metrics,
)


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc5"]

    assert compute_recall_at_k(retrieved, relevant, k=1) == 0.0
    assert compute_recall_at_k(retrieved, relevant, k=2) == 0.5
    assert compute_recall_at_k(retrieved, relevant, k=5) == 1.0
    assert compute_recall_at_k([], relevant, k=5) == 0.0
    assert compute_recall_at_k(retrieved, [], k=5) == 1.0


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3"]

    assert compute_precision_at_k(retrieved, relevant, k=1) == 1.0
    assert compute_precision_at_k(retrieved, relevant, k=2) == 0.5
    assert compute_precision_at_k(retrieved, relevant, k=5) == 0.4
    assert compute_precision_at_k([], relevant, k=3) == 0.0
    # Boundary: when fewer items retrieved than k, denominator must be k
    assert compute_precision_at_k(["doc1"], ["doc1"], k=5) == 0.2


def test_mrr():
    # Hit at rank 1 -> MRR = 1.0
    assert compute_mrr(["doc1", "doc2", "doc3"], ["doc1"]) == 1.0
    # Hit at rank 2 -> MRR = 0.5
    assert compute_mrr(["doc1", "doc2", "doc3"], ["doc2"]) == 0.5
    # Hit at rank 3 -> MRR = 1/3
    assert round(compute_mrr(["doc1", "doc2", "doc3"], ["doc3"]), 4) == round(1.0/3.0, 4)
    # No hit -> MRR = 0.0
    assert compute_mrr(["doc1", "doc2"], ["doc9"]) == 0.0


def test_ndcg_at_k():
    # Perfect ranking: relevant doc at rank 1
    assert compute_ndcg_at_k(["doc1", "doc2", "doc3"], ["doc1"], k=3) == 1.0

    # Suboptimal ranking: relevant doc at rank 2
    # DCG = 1 / log2(3) = 0.6309; IDCG = 1 / log2(2) = 1.0 -> NDCG = 0.6309
    ndcg_rank2 = compute_ndcg_at_k(["doc2", "doc1", "doc3"], ["doc1"], k=3)
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert abs(ndcg_rank2 - expected_ndcg) < 1e-4

    # Miss:
    assert compute_ndcg_at_k(["doc2", "doc3"], ["doc1"], k=2) == 0.0

    # Duplicate chunks from same relevant document must not inflate NDCG > 1.0
    assert compute_ndcg_at_k(["doc1", "doc1", "doc1"], ["doc1"], k=3) == 1.0
    assert compute_ndcg_at_k(["doc2", "doc1", "doc1"], ["doc1"], k=3) == pytest.approx(expected_ndcg, rel=1e-4)


def test_bootstrap_uncertainty():
    vals = [0.8, 0.85, 0.9, 0.95, 0.88, 0.92, 0.86, 0.89]
    res = bootstrap_confidence_interval(vals, n_bootstraps=500, seed=42)

    assert "mean" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]
    assert res["sample_size"] == len(vals)


def test_conflict_detection_metrics():
    preds = [
        {"id": "q1", "detected_conflict": True},
        {"id": "q2", "detected_conflict": False},
        {"id": "q3", "detected_conflict": True},
        {"id": "q4", "detected_conflict": False}
    ]
    gt = [
        {"id": "q1", "has_conflict": True},
        {"id": "q2", "has_conflict": True},
        {"id": "q3", "has_conflict": False},
        {"id": "q4", "has_conflict": False}
    ]
    # TP: q1, FN: q2, FP: q3, TN: q4
    # Precision: 1 / (1+1) = 0.5, Recall: 1 / (1+1) = 0.5
    res = compute_conflict_detection_metrics(preds, gt)
    assert res["conflict_precision"] == 0.5
    assert res["conflict_recall"] == 0.5
    assert res["true_positives"] == 1
    assert res["false_negatives"] == 1
