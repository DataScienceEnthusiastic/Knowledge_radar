"""
===============================================================================
📦 EVALUATION ADAPTER: Multi-Strategy Retriever (Dense, BM25, and RRF Hybrid)
===============================================================================
Provides non-intrusive retrieval adapters around Knowledge Radar:
1. Dense Vector Retrieval (Cosine similarity on normalized feature vectors)
2. Okapi BM25 Lexical Retrieval (k1=1.2, b=0.75, rank-scale term saturation)
3. Reciprocal Rank Fusion (RRF) Hybrid Retrieval (Configurable RRF constant k=60)

Why RRF?
- Rank-scale invariant: Does not require normalizing incompatible score distributions
  (e.g., unbounded BM25 scores vs [0, 1] cosine similarities).
- Zero hyperparameter tuning required across varied query distributions.
- Robust against single-modality outliers and score drift.
===============================================================================
"""

import math
import re
import time
from collections import Counter
from typing import List, Dict, Any, Optional
from modules.vector_store import KnowledgeVectorStore


class BM25Retriever:
    """
    Pure-Python deterministic Okapi BM25 lexical retrieval engine.
    Follows standard BM25 formula with k1=1.2 and b=0.75.
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []          # List of chunk dicts
        self.doc_len = []         # Length in tokens for each chunk
        self.doc_term_freqs = []  # Per-document token count dictionary
        self.avg_doc_len = 0.0
        self.doc_freqs = {}       # Term -> count of documents containing term
        self.idf = {}             # Term -> computed inverse document frequency
        self.num_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', str(text).lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Indexes text chunks for BM25 ranking."""
        self.corpus = chunks
        self.num_docs = len(chunks)
        if self.num_docs == 0:
            return

        self.doc_len = []
        self.doc_freqs = {}
        self.doc_term_freqs = []
        total_tokens = 0

        for chunk in chunks:
            tokens = self._tokenize(chunk.get("text", ""))
            counts = Counter(tokens)
            self.doc_term_freqs.append(counts)
            self.doc_len.append(len(tokens))
            total_tokens += len(tokens)

            for term in counts:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = total_tokens / self.num_docs if self.num_docs > 0 else 1.0

        # Calculate standard smoothed Robertson-Sparck Jones IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF: ln((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Scores all indexed chunks against query using Okapi BM25."""
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = [0.0] * self.num_docs

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for i, counts in enumerate(self.doc_term_freqs):
                tf = counts.get(token, 0)
                if tf == 0:
                    continue

                len_norm = 1.0 - self.b + self.b * (self.doc_len[i] / self.avg_doc_len)
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * len_norm
                scores[i] += idf_val * (num / denom)

        # Rank by score descending
        ranked_indices = sorted(range(self.num_docs), key=lambda i: scores[i], reverse=True)

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], 1):
            if scores[idx] <= 0.0 and rank > 1:
                # Still include top candidates if relevant
                pass
            chunk = dict(self.corpus[idx])
            chunk["bm25_score"] = round(scores[idx], 4)
            chunk["rank"] = rank
            results.append(chunk)

        return results


class HybridRetrieverAdapter:
    """
    Unified Retrieval Adapter supporting Dense, BM25, and Reciprocal Rank Fusion (RRF).
    Wraps existing KnowledgeVectorStore non-destructively.
    """
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.dense_store = KnowledgeVectorStore()
        self.bm25_retriever = BM25Retriever()
        self.raw_chunks = []

    def index_documents(self, chunk_items: List[Dict[str, Any]]):
        """Indexes chunks in both Dense vector store and BM25 index."""
        self.raw_chunks = chunk_items
        self.dense_store.clear_store()
        self.dense_store.add_chunks(chunk_items)
        self.bm25_retriever.index_chunks(chunk_items)

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs pure dense cosine vector retrieval."""
        t0 = time.perf_counter()
        results = self.dense_store.search_chunks(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        for rank, res in enumerate(results, 1):
            res["rank"] = rank
            res["retrieval_strategy"] = "dense"
            res["latency_ms"] = round(elapsed_ms, 3)
        return results

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs pure Okapi BM25 lexical retrieval."""
        t0 = time.perf_counter()
        results = self.bm25_retriever.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        formatted = []
        for rank, item in enumerate(results, 1):
            formatted.append({
                "text": item["text"],
                "source": item.get("source", "Document"),
                "chunk_index": item.get("chunk_index", rank),
                "bm25_score": item.get("bm25_score", 0.0),
                "rank": rank,
                "retrieval_strategy": "bm25",
                "latency_ms": round(elapsed_ms, 3)
            })
        return formatted

    def retrieve_hybrid_rrf(self, query: str, top_k: int = 5, candidate_k: int = 20) -> List[Dict[str, Any]]:
        """
        Executes Reciprocal Rank Fusion (RRF) over Dense and BM25 candidate pools.
        
        RRF Score Formula:
            RRF_score(d) = sum_{m in {dense, bm25}} 1 / (rrf_k + rank_m(d))
        """
        t0 = time.perf_counter()
        dense_candidates = self.dense_store.search_chunks(query, top_k=candidate_k)
        bm25_candidates = self.bm25_retriever.search(query, top_k=candidate_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        doc_scores = {}
        doc_payloads = {}

        # 1. Score dense rankings
        for rank, item in enumerate(dense_candidates, 1):
            key = (item["source"], item["chunk_index"])
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))
            doc_payloads[key] = item

        # 2. Score BM25 rankings
        for rank, item in enumerate(bm25_candidates, 1):
            key = (item.get("source", "Document"), item.get("chunk_index", 0))
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))
            if key not in doc_payloads:
                doc_payloads[key] = item

        # 3. Sort by combined RRF score descending
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)

        results = []
        for rank, key in enumerate(sorted_keys[:top_k], 1):
            item = dict(doc_payloads[key])
            item["rrf_score"] = round(doc_scores[key], 6)
            item["rank"] = rank
            item["retrieval_strategy"] = f"hybrid_rrf_k{self.rrf_k}"
            item["latency_ms"] = round(elapsed_ms, 3)
            results.append(item)

        return results
