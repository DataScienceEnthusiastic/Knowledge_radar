"""
Local Vector Store (Pure-Python Feature Hashing Engine).

Computes deterministic unit-normalized feature embeddings via CRC32 hashing
and performs cosine similarity retrieval without external C++ or ONNX dependencies.
"""

import math
import re
import zlib
from config import COLLECTION_NAME, DEFAULT_TOP_K


class KnowledgeVectorStore:
    """
    Manages high-speed local vector storage and similarity search for Knowledge Radar.
    """
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.chunks = []

    def _embed(self, texts: list) -> list:
        """Generates 256-dimensional feature vectors for input texts."""
        embeddings = []
        dim = 256
        for text in texts:
            tokens = re.findall(r'\w+', str(text).lower())
            vec = [0.0] * dim
            for token in tokens:
                idx = (zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF) % dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings

    def add_chunks(self, chunk_items: list) -> int:
        """
        Indexes text chunks into local vector store.
        
        Args:
            chunk_items (list): List of dicts containing 'text', 'source', 'chunk_index'.
            
        Returns:
            int: Number of chunks indexed.
        """
        if not chunk_items:
            return 0

        texts = [item["text"] for item in chunk_items]
        vecs = self._embed(texts)

        for item, vec in zip(chunk_items, vecs):
            self.chunks.append({
                "text": item["text"],
                "source": item.get("source", "Document"),
                "chunk_index": item.get("chunk_index", len(self.chunks) + 1),
                "embedding": vec
            })
        return len(chunk_items)

    def search_chunks(self, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """
        Queries vector store for top matching text chunks relative to query via cosine similarity.
        When multiple distinct sources are indexed, guarantees balanced representation across
        all sources to prevent any single document from monopolizing the context window.
        """
        if not self.chunks:
            return []

        q_vec = self._embed([query])[0]
        scored = []

        for item in self.chunks:
            dot = sum(a * b for a, b in zip(q_vec, item["embedding"]))
            scored.append((dot, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        unique_sources = list(dict.fromkeys(item["source"] for _, item in scored))

        # Single source or strict top_k smaller than source count
        if len(unique_sources) <= 1 or top_k < len(unique_sources):
            retrieved = []
            for score, item in scored[:top_k]:
                retrieved.append({
                    "text": item["text"],
                    "source": item["source"],
                    "chunk_index": item["chunk_index"],
                    "distance": round(1.0 - score, 4),
                    "score": round(score, 4)
                })
            return retrieved

        # Multi-source balanced retrieval:
        # Guarantee representation for every active document/source in the index
        quota_per_source = max(1, top_k // len(unique_sources))

        source_buckets = {}
        for score, item in scored:
            src = item["source"]
            source_buckets.setdefault(src, []).append((score, item))

        selected_items = []
        selected_ids = set()

        # Step 1: Allocate guaranteed quota per active source
        for src in unique_sources:
            items_for_src = source_buckets.get(src, [])
            for score, item in items_for_src[:quota_per_source]:
                item_id = (item["source"], item["chunk_index"])
                if item_id not in selected_ids and len(selected_items) < top_k:
                    selected_ids.add(item_id)
                    selected_items.append((score, item))

        # Step 2: Fill remaining budget with highest scoring chunks globally
        if len(selected_items) < top_k:
            for score, item in scored:
                item_id = (item["source"], item["chunk_index"])
                if item_id not in selected_ids:
                    selected_ids.add(item_id)
                    selected_items.append((score, item))
                    if len(selected_items) >= top_k:
                        break

        # Step 3: Re-sort selected items by relevance score
        selected_items.sort(key=lambda x: x[0], reverse=True)

        retrieved = []
        for score, item in selected_items[:top_k]:
            retrieved.append({
                "text": item["text"],
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "distance": round(1.0 - score, 4),
                "score": round(score, 4)
            })
        return retrieved

    def get_sources_summary(self) -> dict:
        """Returns a breakdown of chunk counts per unique source."""
        summary = {}
        for item in self.chunks:
            src = item.get("source", "Document")
            summary[src] = summary.get(src, 0) + 1
        return summary

    def clear_store(self):
        """Clears all indexed vector chunks."""
        self.chunks.clear()

    def count(self) -> int:
        """Returns total number of chunks currently indexed."""
        return len(self.chunks)
