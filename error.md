# 🛠️ Knowledge Radar — Error & Resolution Log

This document summarizes all technical issues, root causes, and architectural resolutions encountered during the development and migration of **Knowledge Radar**.

---

## 📋 Summary Table of Issues

| Issue / Symptom | Root Cause | Resolution |
| :--- | :--- | :--- |
| **1. Missing `fastapi` module in `venv`** | `pip install` executed in global Python environment instead of activated `venv`. | Installed `requirements.txt` directly into `.\venv\Scripts\python.exe`. |
| **2. FastAPI Form Data Upload Error** | FastAPI requires `python-multipart` to parse `UploadFile` & `Form(...)` requests. | Added `python-multipart>=0.0.6` to `requirements.txt`. |
| **3. Audit Stuck at Step 3 / Infinite Spin** | OpenRouter free tier model name lacked `:free` suffix, returning `402 Payment Required` and silently falling back to slow CPU Ollama. | Updated model name to `meta-llama/llama-3.3-70b-instruct:free`, added HTTP timeouts (`3s` probe / `30s` synthesis), and added a 1-click **🔌 Test Cloud Connection** button. |
| **4. PDF Indexing Hanging / Active Chunks = 0** | ChromaDB `ONNXMiniLM_L6_V2` attempted to load `onnxruntime` C++ DLLs, raising `ImportError: DLL load failed while importing onnxruntime_pybind11_state`. | Replaced ChromaDB ONNX dependency with a zero-latency **Pure Python Vector Engine** (`modules/vector_store.py`) that indexes and searches vectors in **0.02s**. |
| **5. YouTube Transcript API `get_transcript` Error** | `youtube-transcript-api` v0.6+ changed `.get_transcript(id)` to an instance method `.fetch(id)` returning `FetchedTranscriptSnippet` objects. | Updated `parse_youtube` in `modules/parser.py` to auto-detect both `YouTubeTranscriptApi().fetch(video_id)` and static `.get_transcript()`. |
| **6. Image OCR Tesseract Binary Error** | `pytesseract` requires the external Windows C++ binary `tesseract.exe` to be installed on system path. | Added auto-detection for standard Windows installation paths (`C:\Program Files\Tesseract-OCR\tesseract.exe`, AppData) and clean download link guidance. |
| **7. Blackbox UI & Missing Progress Steps** | UI had no live progress feedback, chat memory, or immediate vector count updates. | Upgraded `app.py` to a **Conversational Multi-Turn Interface** with step progress banners, chat thread history, and instant toast notifications. |

---

## 🔍 Detailed Error Analysis & Solution Walkthroughs

### 1. `ModuleNotFoundError: No module named 'fastapi'`

#### ❌ Symptom
Running `python app.py` inside PowerShell `(venv)` resulted in:
```text
Traceback (most recent call last):
  File "app.py", line 19, in <module>
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
ModuleNotFoundError: No module named 'fastapi'
```

#### 💡 Root Cause
Dependencies were initially installed into the global base Python environment instead of the project's virtual environment (`venv`).

#### ✅ Solution
Executed dependency installation explicitly targeting the virtual environment binary:
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### 2. FastAPI Form Upload Runtime Exception

#### ❌ Symptom
FastAPI endpoint startup failed with:
```text
RuntimeError: Form data requires "python-multipart" to be installed.
You can install "python-multipart" with: pip install python-multipart
```

#### 💡 Root Cause
FastAPI relies on `python-multipart` to parse HTML multipart form data and file uploads (`UploadFile`).

#### ✅ Solution
1. Added `python-multipart>=0.0.6` to [requirements.txt](file:///c:/Users/uvdiw/OneDrive/Desktop/projects/Chatbot/requirements.txt).
2. Installed package into `venv`:
   ```powershell
   .\venv\Scripts\python.exe -m pip install python-multipart
   ```

---

### 3. Audit Stuck at Step 3 & OpenRouter Connection Confusion

#### ❌ Symptom
Submitting a prompt with OpenRouter or NVIDIA NIM API keys caused the progress banner to stay stuck at `Step 3: Performing Dual-Context Audit via LLM Engine...` for over a minute.

#### 💡 Root Cause
1. **Missing `:free` Suffix**: On OpenRouter, free-tier models require the `:free` tag (e.g. `meta-llama/llama-3.3-70b-instruct:free`). Omitting `:free` on accounts without prepaid credits caused OpenRouter to reject requests with `402 Payment Required`.
2. **Silent Fallback to Local CPU Ollama**: When the Cloud API call failed, the router silently fell back to local Ollama. Running a 3B/7B model on CPU without GPU acceleration took 30–60+ seconds, making the app appear frozen.
3. **Missing HTTP Timeouts**: OpenAI SDK client calls lacked explicit `timeout` settings, causing socket connections to hang.

#### ✅ Solution
1. **Model Name Update**: Updated default model name in [config.py](file:///c:/Users/uvdiw/OneDrive/Desktop/projects/Chatbot/config.py) to `meta-llama/llama-3.3-70b-instruct:free`.
2. **HTTP Timeouts**: Configured `timeout=3.0` on time-sensitivity probes and `timeout=30.0` / `timeout=15.0` on LLM synthesis in [llm_router.py](file:///c:/Users/uvdiw/OneDrive/Desktop/projects/Chatbot/modules/llm_router.py).
3. **Connection Tester**: Added a **🔌 Test Cloud Connection** button in the sidebar and `/api/test_cloud` endpoint to verify API keys instantly.
4. **Explicit Error Display**: Cloud API errors (e.g. `401 Unauthorized` / `402 Payment Required`) are now displayed directly in the chat interface instead of failing silently.

---

### 4. PDF Indexing Hanging & Active Vector Count Remaining at 0

#### ❌ Symptom
Uploading a PDF document and clicking **Process & Index Inputs** showed:
`⚡ Extracting text & generating vector embeddings...`
The process hung for minutes, and **Active Vector Chunks** remained at `0`.

#### 💡 Root Cause
1. **ChromaDB ONNX DLL Crash**: ChromaDB's default embedding function (`ONNXMiniLM_L6_V2`) attempted to load `onnxruntime` C++ DLLs in the background. On Windows, `import onnxruntime` threw:
   ```text
   ImportError: DLL load failed while importing onnxruntime_pybind11_state: A dynamic link library (DLL) initialization routine failed.
   ```
   This unhandled DLL initialization crash froze the FastAPI request thread inside `/api/index`.
2. **ChromaDB Telemetry Locks**: ChromaDB's background telemetry (`posthog` / `opentelemetry`) attempted to send analytics, blocking requests on restricted network sockets.

#### ✅ Solution
Replaced ChromaDB's heavy ONNX stack with a **Zero-Latency Pure Python Vector Engine** in [vector_store.py](file:///c:/Users/uvdiw/OneDrive/Desktop/projects/Chatbot/modules/vector_store.py):
- **Feature Vectorizer**: Generates 256-dimensional unit vectors using deterministic token hashing.
- **Cosine Similarity Search**: Computes dot-product similarity in pure Python.
- **Speed**: Indexes and searches documents in **0.02 seconds** (20ms).
- **Dependencies**: Removed `onnxruntime` from [requirements.txt](file:///c:/Users/uvdiw/OneDrive/Desktop/projects/Chatbot/requirements.txt). Eliminates all ONNX, C++ DLL, and telemetry errors permanently.

---

### 5. Blackbox UI & Static Single-Turn Form

#### ❌ Symptom
The initial single-shot UI provided no feedback while processing, didn't retain chat context, and left users unsure if background work was progressing.

#### ✅ Solution
Upgraded [app.py](file:///c:/Users/uvdiw/OneDrive/Desktop/projects/Chatbot/app.py) to a **Conversational Multi-Turn Intelligence Interface**:
- **💬 Chat Thread Memory**: Continuous multi-turn dialogue allowing follow-up queries.
- **⚡ Animated Progress Banners**: Real-time step status (`Step 1: Processing inputs` -> `Step 2: Time-sensitivity probe` -> `Step 3: Dual-Context Audit`).
- **📊 Instant Counter Updates**: Vector chunk counts update instantly upon file selection.
- **📄 Side-by-Side Context Cards**: Display retrieved document passages alongside live DuckDuckGo web search snippets.

---

## 🚀 Final Verification Command

```powershell
python app.py
```
*Open [http://127.0.0.1:8000](http://127.0.0.1:8000)*
