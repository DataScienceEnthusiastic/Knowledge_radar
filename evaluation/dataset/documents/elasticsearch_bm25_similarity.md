# Elasticsearch BM25 Scoring and Index Sharding

Elasticsearch ranks search hits using the Okapi BM25 scoring algorithm, which models term frequency saturation and inverse document frequency (IDF). Parameters $k_1$ (typically 1.2) controls term frequency scaling, while $b$ (typically 0.75) controls document length normalization. Indices are divided into primary and replica shards distributed across cluster nodes. Search requests execute in a two-phase Query-Then-Fetch process: shards return matching doc IDs and scores, followed by fetching full source documents.
