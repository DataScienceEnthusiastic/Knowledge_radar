"""
===============================================================================
📦 EVALUATION ADAPTER: Candidate Pool Reranker
===============================================================================
Evaluates candidate pool reranking (Top-10 -> 5 or Top-20 -> 5) against direct
retrieval. Measures both retrieval quality gains and compute latency overheads.
===============================================================================
"""

import math
import re
import time
from typing import List, Dict, Any


class CrossScoreReranker:
    """
    Evaluates candidate chunks against query using fine-grained token-level cross scoring:
    1. Exact query n-gram sequence match bonus
    2. Term co-occurrence density
    3. Positional proximity weighting
    """
    def __init__(self, phrase_weight: float = 2.0, proximity_weight: float = 1.5):
        self.phrase_weight = phrase_weight
        self.proximity_weight = proximity_weight

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', str(text).lower())

    def score_pair(self, query: str, document_text: str) -> float:
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(document_text)

        if not q_tokens or not d_tokens:
            return 0.0

        q_set = set(q_tokens)
        d_set = set(d_tokens)
        overlap = q_set.intersection(d_set)

        if not overlap:
            return 0.0

        # Base token coverage
        coverage = len(overlap) / len(q_set)

        # Exact bigram/trigram phrase matching bonus
        phrase_bonus = 0.0
        q_text_lower = query.lower()
        d_text_lower = document_text.lower()

        for i in range(len(q_tokens) - 1):
            bigram = f"{q_tokens[i]} {q_tokens[i+1]}"
            if bigram in d_text_lower:
                phrase_bonus += 0.5

        # Term proximity in document text
        token_positions = [i for i, t in enumerate(d_tokens) if t in overlap]
        proximity_bonus = 0.0
        if len(token_positions) > 1:
            span = token_positions[-1] - token_positions[0] + 1
            proximity_bonus = min(1.0, len(token_positions) / max(span, 1))

        total_score = coverage + (self.phrase_weight * phrase_bonus) + (self.proximity_weight * proximity_bonus)
        return total_score

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        """
        Reranks a candidate list and returns top_k results with timing metadata.
        """
        t0 = time.perf_counter()

        scored = []
        for item in candidates:
            score = self.score_pair(query, item.get("text", ""))
            scored.append((score, item))

        # Sort by reranker score descending, tie-break on original rank
        scored.sort(key=lambda x: (x[0], -x[1].get("rank", 999)), reverse=True)

        rerank_latency_ms = (time.perf_counter() - t0) * 1000.0

        reranked_results = []
        for new_rank, (score, orig_item) in enumerate(scored[:top_k], 1):
            res = dict(orig_item)
            res["rerank_score"] = round(score, 4)
            res["orig_rank"] = orig_item.get("rank", new_rank)
            res["rank"] = new_rank
            reranked_results.append(res)

        return {
            "results": reranked_results,
            "rerank_latency_ms": round(rerank_latency_ms, 3),
            "candidate_pool_size": len(candidates),
            "output_k": top_k
        }
