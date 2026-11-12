# 📐 Structured Design Decisions Log — Knowledge Radar

> **Governance Standard**: Every major architecture and algorithmic choice in Knowledge Radar is grounded in empirical experimental data. No design choice is adopted without a recorded tradeoff log.

---

### DECISION-001: Document Chunking Size & Overlap Ratio
- **Problem**: Fixed-size chunking can either dilute factual density if too large or truncate syntactic context if too small.
- **Hypothesis**: A 500-character chunk with a 10% overlap (50 chars) will optimize the trade-off between retrieval recall and context token footprint.
- **Alternatives Considered**:
  1. Ultra-small chunks: 256 chars (0%, 10%, 20% overlap).
  2. Medium chunks: 500 chars (0%, 10%, 20% overlap).
  3. Large chunks: 750 / 1000 chars (0%, 10%, 20% overlap).
- **Experiment Conducted**: `EXP-01` (Chunking Factorial Optimization on Validation Set $N=50$).
- **Empirical Evidence**:
  - Chunk size 500 (10% overlap) achieved **Recall@1 = 0.6300**, **Recall@5 = 0.7700**, **MRR = 0.7247**, and **NDCG@5 = 0.7208** at **1.96 ms** latency with 117 total chunks.
  - Chunk size 256 increased total chunks to 187 (+60%) with higher latency (3.35 ms) without proportional gains in Recall@1 (0.63 vs 0.63).
  - Chunk sizes 750 and 1000 produced fewer chunks (59–64) but aggregated unrelated sentences into the same chunk, increasing context noise.
- **Decision Taken**: Standardize on **500 characters with 50 characters overlap (10%)**, with **Sentence-Aware splitting** for grammatical preservation.
- **Tradeoffs Accepted**: Slightly lower document-level recall than 1000-character chunks, but significantly cleaner evidence boundaries for downstream synthesis.
- **Confidence Level**: **High**.
- **What Would Invalidate This Decision**: Ingesting extensive legal contracts or source code repositories requiring multi-page unbroken AST parse trees.

---

### DECISION-002: Primary Retrieval Architecture — Reciprocal Rank Fusion (RRF)
- **Problem**: Dense vector embeddings struggle on exact technical acronyms and rare entity codes, while pure BM25 struggles with paraphrasing and semantic queries.
- **Hypothesis**: Combining Dense Cosine Search and Okapi BM25 via Reciprocal Rank Fusion (RRF $k=60$) will provide superior consistency across heterogeneous query categories.
- **Alternatives Considered**:
  1. Dense vector retrieval only.
  2. Pure Okapi BM25 lexical retrieval only.
  3. Linear score interpolation: $\alpha \cdot Dense + (1-\alpha) \cdot BM25$.
  4. Reciprocal Rank Fusion (RRF $k=60$).
- **Experiment Conducted**: `EXP-02` (Retrieval Baselines across 7 categories, $N=50$).
- **Empirical Evidence**:
  - Dense retrieval scored poorly on semantic (Recall@5 = 0.3750) and entity-specific (0.6667).
  - BM25 scored high on exact matches (Recall@5 = 0.9800) at fast speeds (**0.52 ms** with Counter-cached indexing).
  - Hybrid RRF achieved **Recall@5 = 0.9100**, **MRR = 0.8307**, and **NDCG@5 = 0.8382**, successfully lifting dense failures in factual, entity-specific, and multi-hop queries without requiring score normalization.
- **Decision Taken**: Implement **Reciprocal Rank Fusion ($k=60$)** as the default multi-modal retrieval engine.
- **Tradeoffs Accepted**: Retrieval latency is slightly higher (5.65 ms) than dense-only (1.97 ms), but well within the sub-10ms real-time budget.
- **Confidence Level**: **High**.
- **What Would Invalidate This Decision**: Moving to a unified 1024-D fine-tuned bi-encoder embedding model that achieves >95% recall across both semantic and exact lexical entities.

---

### DECISION-003: Candidate Pool Reranking Depth
- **Problem**: Passing all candidate chunks to an LLM inflates prompt costs and triggers Lost-in-the-Middle degradation. Direct Top-5 retrieval may miss relevant documents ranked at positions 6–10.
- **Hypothesis**: Extracting a shallow candidate pool of Top-10 chunks and reranking to Top-5 via cross-scoring will improve NDCG without latency penalties.
- **Alternatives Considered**:
  1. Direct Top-5 retrieval (No reranker).
  2. Candidate Pool Top-10 $\rightarrow$ Rerank $\rightarrow$ Context Top-5.
  3. Candidate Pool Top-20 $\rightarrow$ Rerank $\rightarrow$ Context Top-5.
- **Experiment Conducted**: `EXP-03` (Reranking Depth Tradeoff on Validation Set $N=50$).
- **Empirical Evidence**:
  - **Direct Top-5 (No Rerank)**: **NDCG@5 = 0.8628**, **MRR = 0.8373**, **Recall@5 = 0.9800**, Total Latency = **5.01 ms**.
  - Top-10 $\rightarrow$ Top-5: NDCG@5 = 0.8309, MRR = 0.8063, Recall@5 = 0.9300, Rerank Latency = 0.77 ms, Total Latency = 6.34 ms.
  - Top-20 $\rightarrow$ Top-5: NDCG@5 degraded to 0.7986 and MRR to 0.7767 as distractor candidates diluted the lexical reranker.
- **Decision Taken**: Standardize on **Direct Top-5 Hybrid RRF retrieval** for primary serving. Keep the cross-scoring reranker available for environments pairing full deep cross-encoders.
- **Tradeoffs Accepted**: Avoiding superficial token-overlap reranking prevents promoting distractor chunks over true semantic context.
- **Confidence Level**: **High**.
- **What Would Invalidate This Decision**: Adding a heavy deep-learning cross-encoder (e.g. BGE-Reranker-Large) with >200ms CPU inference times.

---

### DECISION-004: Adaptive Conditional Web Verification
- **Problem**: Live web search is essential for real-time freshness and price changes, but searching on every query introduces latency, rate limits, and irrelevant noise.
- **Hypothesis**: A fast time-sensitivity heuristic probe will route temporal queries to external search while keeping static queries insulated from web noise.
- **Alternatives Considered**:
  1. Mode A: Web search completely disabled (User documents only).
  2. Mode B: Unconditional web search on every query.
  3. Mode C: Adaptive conditional web search.
- **Experiment Conducted**: `EXP-04` (Web Verification & Conflict Audit, $N=50$).
- **Empirical Evidence**:
  - Mode B (Always Web) injected 9.6 extraneous noise tokens per query on static questions.
  - Mode C (Adaptive) achieved **100% Temporal Accuracy** and **100% Conflict Recall** while reducing extraneous static noise to **1.3 tokens (an 86.5% reduction)**.
- **Decision Taken**: Implement **Adaptive Conditional Web Routing** as the core verification policy.
- **Tradeoffs Accepted**: Requires maintaining a lightweight keyword probe and caching server connectivity state.
- **Confidence Level**: **High**.
- **What Would Invalidate This Decision**: Queries where temporal sensitivity is completely implicit without any temporal or market keywords.

---

### DECISION-005: Hybrid LLM Routing (Cloud Primary + Local Ollama Fallback)
- **Problem**: Cloud LLMs provide high-quality synthesis but are subject to rate limits (HTTP 429), authorization failures, or internet outages.
- **Hypothesis**: A seamless automatic fallback to a local Ollama model (`qwen2.5-coder:1.5b` or `3b`) ensures 100% system availability.
- **Alternatives Considered**:
  1. Cloud-only API gateway (fails hard on network drop or rate limits).
  2. Local-only CPU model (lower synthesis capacity on complex multi-hop comparisons).
  3. Dual-tier fallback routing with cached health checks.
- **Decision Taken**: Implement **Dual-Tier Hybrid Routing** with transparent provider badges displayed in the UI.
- **Tradeoffs Accepted**: Slight variance in output phrasing when cloud quota exhausts and local fallback activates.
- **Confidence Level**: **High**.

---

### DECISION-006: Rejection of GraphRAG and Complex Agentic Frameworks (For Current Scale)
- **Problem**: Over-engineering RAG systems with graph databases, multi-agent frameworks (LangGraph/AutoGen), or fine-tuning before diagnosing the core retrieval pipeline.
- **Empirical Justification**: 
  - `EXP-06A` demonstrated that the hybrid retrieval + reranker pipeline achieved **97.86% Recall@5** on the held-out benchmark.
  - `EXP-06B` error analysis showed that only **2.63%** of failures were retrieval-related, while **55.26%** were LLM reasoning/synthesis limitations.
- **Decision Taken**: Maintain a clean, modular, zero-bloat architecture without GraphRAG or agents.
- **What Would Invalidate This Decision**: Demonstrating a failure class where document knowledge graph traversal (e.g. 5+ entity hops across 10 disjoint documents) cannot be retrieved via RRF hybrid search.
