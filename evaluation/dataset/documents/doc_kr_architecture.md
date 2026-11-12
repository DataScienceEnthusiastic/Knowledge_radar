# Knowledge Radar System Architecture Specification (v2.4)
Knowledge Radar is a zero-latency, local-first multi-modal retrieval-augmented generation platform.
The architecture combines a fast 256-dimensional feature projection vector engine with hybrid retrieval.
In Version 2.4, the default maximum context window was set to 8192 tokens.
The vector index operates in-memory with sub-millisecond cosine similarity search using L2-normalized token embeddings.
Dual-Context Auditing is executed by comparing local retrieval snippets against external verification sources.
The LLM router uses an adaptive rule: queries matching temporal keywords (such as 'latest', 'current', '2025', 'price') trigger real-time web verification.
The cloud synthesis model default is meta-llama/llama-3.3-70b-instruct with local fallback to qwen2.5-coder:3b running on Ollama port 11434.
The primary chunking parameter is 500 characters with 50 characters overlap (10% ratio).
