# ⚠️ System Boundaries, Edge Cases & Limitations — Knowledge Radar

> **Scientific Transparency**: Every engineering system operates within defined empirical boundaries. This document outlines the architectural and operational limitations observed during benchmark testing.

---

## 1. Retrieval & Storage Boundaries

### 1.1 In-Memory Vector Store Scale Ceiling
- **Current Design**: In-memory pure Python feature hashing vector engine.
- **Observed Boundary**: Works with sub-millisecond query latency ($< 3\text{ ms}$) on corpora up to **10,000 document chunks** (~5–10 MB of text).
- **Scaling Limit**: For enterprise corpora exceeding 100,000 chunks, memory usage and linear $O(N)$ dot-product scanning will degrade latency beyond 100ms.
- **Recommended Evolution Path**: Migrate to HNSW graph indexing via embedded SQLite-vec or FAISS/Qdrant when scaling beyond 50,000 chunks.

### 1.2 Fixed Character Chunking
- **Current Design**: Sliding character window (500 chars, 50 chars overlap).
- **Limitation**: Can split sentences across boundary thresholds or divide tabular columns into fragmented tokens.
- **Future Direction**: Implement semantic sentence-boundary chunking via `nltk` or recursive AST-aware syntax tree splitting for code files.

---

## 2. Ingestion & Multi-Modal Processing Boundaries

### 2.1 Tesseract OCR on Complex Documents
- **Current Design**: `pytesseract` with default page segmentation mode (PSM 3).
- **Limitation**: Degrades on multi-column academic layouts, complex financial tables, low-contrast scans, or handwritten notes.
- **Edge Case**: Scanned PDF pages with embedded watermarks can introduce OCR noise into indexed chunks.

### 2.2 YouTube Transcripts
- **Current Design**: `youtube-transcript-api` without audio diarization.
- **Limitation**: Videos lacking closed captions (or with auto-generated captions containing severe phoneme errors) cannot be accurately indexed.
- **Evolution Path**: Add local Whisper (`whisper-base` / `whisper-small`) audio transcription for videos without native transcripts.

---

## 3. Web Verification & Live Search Boundaries

### 3.1 Rate Limiting on External Search
- **Current Design**: DuckDuckGo search (`duckduckgo_search`).
- **Limitation**: Excessive burst query rates (>30 queries/min) can trigger upstream DuckDuckGo rate limiting or HTTP 429 warnings.
- **Mitigation Implemented**: Exponential backoff retry with deterministic fallback to cached snapshots.

### 3.2 Time-Sensitivity Probe False Negatives
- **Current Design**: Hybrid keyword rule + local LLM probe.
- **Limitation**: Implicit temporal questions (e.g. *"Who is the CEO of Twitter?"* without mentioning the year or 'current') may fail to trigger web search unless explicitly phrased with temporal markers.

---

## 4. Compute & Local LLM Synthesis Tradeoffs

### 4.1 8GB RAM Laptop Operational Envelope
- **Sweet Spot**: `qwen2.5-coder:1.5b` or `qwen2.5-coder:3b` running on Ollama CPU inference.
- **Limitation**: 3B models on CPU achieve ~20–28 tokens/second, which introduces 3–5 seconds of synthesis latency for long 300-word dual-audit summaries.
- **Guidance**: Use cloud synthesis (`meta-llama/llama-3.3-70b-instruct`) for interactive low-latency production use, reserving Ollama for offline fallback.
