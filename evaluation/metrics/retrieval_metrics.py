"""
===============================================================================
📊 EVALUATION METRICS: Information Retrieval Metrics (Recall@K, MRR, NDCG@K, Precision@K)
===============================================================================
Standard deterministic IR evaluation functions following TREC / BEIR methodology.
===============================================================================
"""

import math
from typing import List, Set, Any


def compute_recall_at_k(retrieved_items: List[Any], relevant_items: List[Any], k: int) -> float:
    """
    Computes Recall@K: Proportion of relevant items retrieved in top-k results.
    Recall@K = |Retrieved@K intersect Relevant| / |Relevant|
    """
    if not relevant_items:
        return 1.0
    if not retrieved_items or k <= 0:
        return 0.0

    retrieved_at_k = set(retrieved_items[:k])
    relevant_set = set(relevant_items)
    intersection = retrieved_at_k.intersection(relevant_set)

    return float(len(intersection) / len(relevant_set))


def compute_precision_at_k(retrieved_items: List[Any], relevant_items: List[Any], k: int) -> float:
    """
    Computes Precision@K: Proportion of top-k retrieved items that are relevant.
    Precision@K = |Retrieved@K intersect Relevant| / k
    """
    if k <= 0:
        return 0.0
    if not retrieved_items or not relevant_items:
        return 0.0

    retrieved_at_k = set(retrieved_items[:k])
    relevant_set = set(relevant_items)
    intersection = retrieved_at_k.intersection(relevant_set)

    return float(len(intersection) / k)


def compute_mrr(retrieved_items: List[Any], relevant_items: List[Any]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR): 1 / rank of the FIRST relevant item found.
    MRR = 1 / first_hit_rank (or 0.0 if no relevant items in retrieved list).
    """
    if not relevant_items or not retrieved_items:
        return 0.0

    relevant_set = set(relevant_items)
    for rank, item in enumerate(retrieved_items, 1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_items: List[Any], relevant_items: List[Any], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    DCG@K = sum_{i=1}^k (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(|Rel|, k)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K
    """
    if not relevant_items or not retrieved_items or k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    dcg = 0.0
    seen_docs = set()

    for i, item in enumerate(retrieved_items[:k], 1):
        if item in relevant_set and item not in seen_docs:
            dcg += 1.0 / math.log2(i + 1)
            seen_docs.add(item)

    # Ideal DCG calculation
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)
