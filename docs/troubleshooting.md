# Engineering Troubleshooting & Common Resolutions

This document catalogues technical failure modes, root causes, and verified engineering resolutions for Knowledge Radar.

---

## Issue Resolution Index

| Symptom / Error | Root Cause | Engineering Resolution |
| :--- | :--- | :--- |
| **Missing `fastapi` or `openai` module** | Global Python environment invoked instead of the virtual environment (`.venv`). | Invoke `.\.venv\Scripts\python.exe` or activate `.venv` before running scripts. |
| **FastAPI Form Data Upload Error (`python-multipart`)** | FastAPI requires `python-multipart` to parse `UploadFile` & `Form(...)` requests. | Added `python-multipart>=0.0.6` to `requirements.txt`. |
| **Cloud Audit Timeout / Infinite Spin** | OpenRouter free-tier models require `:free` suffix; lacking explicit HTTP timeouts caused thread hangs. | Set model to `meta-llama/llama-3.3-70b-instruct:free`, added 3s/30s HTTP timeouts, and added `/api/test_cloud` connection probe. |
| **Vector Indexing Freeze on Windows (`onnxruntime` DLL)** | ChromaDB's default ONNX backend failed to load C++ runtime DLLs under Windows AppLocker/security policies. | Built a zero-dependency pure Python vector engine (`modules/vector_store.py`) with deterministic hashing. |
| **YouTube Transcript API Method Signature Change** | `youtube-transcript-api` v0.6+ replaced `.get_transcript()` with `.fetch()` returning snippet objects. | Updated `parse_youtube` in `modules/parser.py` to auto-detect both `.fetch()` and `.get_transcript()`. |
| **Pytesseract Binary Not Found on PATH** | `tesseract.exe` is a system binary and not bundled with Python pip packages. | Added Windows auto-detection for standard paths (`C:\Program Files\Tesseract-OCR\tesseract.exe`) and AppData directories. |
| **NDCG@5 Values Exceeding 1.0 (e.g. 1.29–1.39)** | Evaluator credited multiple chunks from the same relevant document as separate positive hits in DCG calculation. | Added deduplication tracking (`seen_docs`) in `evaluation/metrics/retrieval_metrics.py`. |
| **Precision@K Denominator Truncation** | Evaluator divided by `min(k, len(retrieved))` rather than standard $k$, artificially inflating small results. | Replaced denominator with strict $k$ per standard TREC information retrieval evaluation protocols. |
| **Slow BM25 Latency (~14.8 ms/query)** | Document term frequencies re-tokenized inside the query token loop on every search. | Pre-calculated and cached `doc_term_freqs` with `collections.Counter` upon indexing, reducing search latency to 0.52 ms. |

---

## Detailed Issue Analysis

### 1. Python Environment Isolation (`ModuleNotFoundError`)
When executing commands via terminal, PowerShell may default to the system Python in `AppData` rather than the project virtual environment.
To ensure the correct isolated packages are used:
```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe evaluation/experiments/run_all.py
```

### 2. Pure Python Vector Store Architecture
To avoid C++ dynamic link library conflicts (such as `onnxruntime_pybind11_state.pyd` initialization failures on locked Windows corporate environments), Knowledge Radar implements a zero-dependency vector engine:
- **Projection**: 256-dimensional unit feature vectors via deterministic CRC32 token hashing.
- **Distance Calculation**: Pure Python L2-normalized cosine similarity.
- **Latency**: Sub-2ms search latency across corpus vaults.

### 3. OpenRouter Free-Tier Model Configuration
When using OpenRouter without prepaid balance, model identifiers must explicitly specify the `:free` suffix:
- Target identifier: `meta-llama/llama-3.3-70b-instruct:free`
- Router fallback: Gracefully switches to local Ollama (`qwen2.5-coder:1.5b` or `qwen2.5-coder:3b`) on HTTP 401, 402, 429, or network disconnect.

### 4. Sentence-Aware Chunking Boundary Preservation
Blind character-window slicing (`start += 500 - 50`) can truncate tokens, numbers, and dates (e.g., splitting `142 microseconds` into `14` and `2 microseconds`).
Use `chunk_sentences` from `modules/parser.py` to preserve complete grammatical units while respecting chunk size constraints.

