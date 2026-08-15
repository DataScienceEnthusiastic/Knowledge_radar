# 📡 Knowledge Radar

> A zero-cost, live-syncing multi-modal intelligence platform and alternative to NotebookLM featuring **Dual-Context Audit** and **Hybrid Cloud + Local AI Routing**.

Knowledge Radar processes multiple input formats (PDFs, YouTube video transcripts, Web URLs, and Images), stores their vector embeddings locally in ChromaDB, and performs Dual-Context Audits using a hybrid LLM engine with Cloud primary (OpenRouter / NVIDIA NIM) and automatic local Ollama fallback.

---

## 🚀 Features

- **📂 Multi-Modal Processing**:
  - **📄 PDFs**: Text extraction & page chunking using `pypdf`.
  - **🎥 YouTube Videos**: Transcript extraction with `[MM:SS]` timestamps using `youtube-transcript-api`.
  - **🌐 Web URLs**: Web article content extraction via `requests` + `beautifulsoup4`.
  - **🖼️ Images**: Local OCR text extraction using `pytesseract` + `Pillow`.
- **📦 Local Vector Indexing**: Overlapping chunking (500 chars) indexed locally in ChromaDB without high overhead.
- **🤖 Hybrid LLM Engine**:
  - **Local AI (Ollama)**: Automatically probes query time-sensitivity and serves as a seamless fallback.
  - **Cloud LLM (OpenRouter / NVIDIA NIM)**: High-capacity dual-context synthesis; automatically falls back to local Ollama on cloud rate limits (429) or API key omissions.
- **🌐 Real-Time Web Verification**: Executes DuckDuckGo web searches for time-sensitive prompts to cross-verify document facts.
- **⚖️ Split-View Audit & Badging**: Side-by-side display of multi-modal context vs. live web snippets, complete with provider status badges (`[Powered by NVIDIA Cloud]` vs `[Powered by Local Ollama]`).

---

## 📂 Architecture

```
Chatbot/
├── config.py                 # Central settings, model defaults, API URLs & chunk parameters
├── modules/
│   ├── __init__.py           # Package marker
│   ├── parser.py             # PDF, YouTube, Web URL, Image OCR parsers + chunker
│   ├── vector_store.py       # ChromaDB vector collection operations
│   ├── web_search.py         # DuckDuckGo live web search integration
│   └── llm_router.py         # Hybrid LLM Router (Cloud Primary -> Local Ollama Fallback)
├── app.py                    # FastAPI Backend & Single-Page Web Dashboard UI
└── requirements.txt          # Project dependencies
```

---

## 🛠️ Quick Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional for Local AI) Launch Ollama**:
   ```bash
   ollama serve
   ```
   *(Ensure model is pulled, e.g., `ollama pull qwen2.5-coder:3b` or `ollama pull llama3.2:3b`)*

3. **Launch Application**:
   ```bash
   python app.py
   ```
   *Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.*

4. **API Keys (Optional for Cloud LLMs)**:
   - [OpenRouter API Keys](https://openrouter.ai/keys)
   - [NVIDIA NIM API Keys](https://build.nvidia.com/)
