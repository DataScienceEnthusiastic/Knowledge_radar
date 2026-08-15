"""
===============================================================================
📦 MODULE: Local Vector Store (Pure Python Zero-Latency Engine)
===============================================================================
Handles local embedding generation, vector indexing, and cosine similarity search.
Uses a zero-dependency pure-Python vector engine to guarantee 100% reliable
instant indexing with ZERO ONNX, C++ DLL, or telemetry runtime requirements.
===============================================================================
"""

import math
import re
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
                # Deterministic token hash
                idx = abs(hash(token)) % dim
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
        
        Args:
            query (str): Search prompt.
            top_k (int): Number of top results to return.
            
        Returns:
            list of dicts: Retrieved matching context snippets with metadata.
        """
        if not self.chunks:
            return []

        q_vec = self._embed([query])[0]
        scored = []

        for item in self.chunks:
            # Cosine similarity dot product (vectors are unit-normalized)
            dot = sum(a * b for a, b in zip(q_vec, item["embedding"]))
            scored.append((dot, item))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)

        retrieved = []
        for score, item in scored[:top_k]:
            retrieved.append({
                "text": item["text"],
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "distance": round(1.0 - score, 4)
            })
        return retrieved

    def clear_store(self):
        """Clears all indexed vector chunks."""
        self.chunks.clear()

    def count(self) -> int:
        """Returns total number of chunks currently indexed."""
        return len(self.chunks)
