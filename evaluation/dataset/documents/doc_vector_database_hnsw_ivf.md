# Vector Indexing Algorithms: HNSW vs. IVF-PQ Tradeoffs
Hierarchical Navigable Small World (HNSW) constructs a multi-layer graph where bottom layers contain all vectors and top layers contain sparse links.
Key parameters: M (number of bidirectional links per vector, typically 16-64) and efConstruction (size of dynamic candidate list, typically 100-200).
HNSW provides high recall (>98% at top-10) with sub-10ms query latency, but incurs 1.5x to 2.0x memory overhead for graph pointers.
Inverted File with Product Quantization (IVF-PQ) compresses vectors into compact byte codes using k-means centroid codebooks.
IVF-PQ reduces RAM consumption by 80-90% but degrades recall by 5-15% on high-dimensional vectors compared to uncompressed HNSW.
