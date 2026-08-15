"""
===============================================================================
📡 KNOWLEDGE RADAR — Conversational Multi-Modal Intelligence Platform
===============================================================================
FastAPI + Modern Interactive Single-Page Conversational Dashboard.
Processes PDFs, YouTube videos, Web URLs, and Images into a local ChromaDB store
and performs Dual-Context Audits using hybrid Cloud + Local Ollama routing.

Launch Command:
    python app.py
===============================================================================
"""

import os
import io
import tempfile
import uvicorn
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import (
    LOCAL_MODEL_NAME, CLOUD_MODEL_NAME, OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL, NVIDIA_NIM_BASE_URL, DEFAULT_TOP_K
)
from modules.parser import parse_pdf, parse_youtube, parse_url, parse_image, chunk_text
from modules.vector_store import KnowledgeVectorStore
from modules.web_search import fetch_live_web_context
from modules.llm_router import check_ollama_online, is_time_sensitive_local, generate_response, test_cloud_connection

app = FastAPI(title="Knowledge Radar API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton vector store instance
vector_store = KnowledgeVectorStore()
indexed_signatures = set()


def auto_index_inputs(
    pdf_bytes: Optional[bytes] = None,
    pdf_filename: str = "",
    img_bytes: Optional[bytes] = None,
    img_filename: str = "",
    yt_url: str = "",
    web_url: str = ""
):
    """
    Processes and indexes provided file bytes or URLs into ChromaDB if not already indexed.
    """
    global indexed_signatures
    new_chunks = []
    messages = []

    # 1. Process PDF
    if pdf_bytes and pdf_filename:
        sig = f"pdf_{pdf_filename}_{len(pdf_bytes)}"
        if sig not in indexed_signatures:
            pdf_file_obj = io.BytesIO(pdf_bytes)
            text = parse_pdf(pdf_file_obj)
            if text and not text.startswith("Error"):
                chunks = chunk_text(text, source_name=f"PDF: {pdf_filename}")
                if chunks:
                    new_chunks.extend(chunks)
                    indexed_signatures.add(sig)
                    messages.append(f"Parsed PDF '{pdf_filename}' ({len(chunks)} chunks)")
                else:
                    messages.append(f"PDF '{pdf_filename}' contains no extractable plain text. (If scanned, convert pages to images for OCR).")
            else:
                err_detail = text if text else "No text extracted from PDF."
                messages.append(f"PDF Error: {err_detail}")

    # 2. Process Image OCR
    if img_bytes and img_filename:
        sig = f"img_{img_filename}_{len(img_bytes)}"
        if sig not in indexed_signatures:
            img_file_obj = io.BytesIO(img_bytes)
            text = parse_image(img_file_obj)
            if text and not text.startswith("Error"):
                chunks = chunk_text(text, source_name=f"Image: {img_filename}")
                if chunks:
                    new_chunks.extend(chunks)
                    indexed_signatures.add(sig)
                    messages.append(f"Parsed Image '{img_filename}' ({len(chunks)} chunks)")
                else:
                    messages.append(f"Image '{img_filename}' yielded no OCR text.")
            else:
                messages.append(f"Image Error: {text}")

    # 3. Process YouTube URL
    if yt_url and yt_url.strip():
        url_clean = yt_url.strip()
        sig = f"yt_{url_clean}"
        if sig not in indexed_signatures:
            text = parse_youtube(url_clean)
            if text and not text.startswith("Error"):
                chunks = chunk_text(text, source_name=f"YouTube ({url_clean[:30]}...)")
                if chunks:
                    new_chunks.extend(chunks)
                    indexed_signatures.add(sig)
                    messages.append(f"Parsed YouTube Transcript ({len(chunks)} chunks)")
                else:
                    messages.append(f"YouTube video produced 0 transcript chunks.")
            else:
                messages.append(f"YouTube Error: {text}")

    # 4. Process Web URL
    if web_url and web_url.strip():
        url_clean = web_url.strip()
        sig = f"web_{url_clean}"
        if sig not in indexed_signatures:
            text = parse_url(url_clean)
            if text and not text.startswith("Error"):
                chunks = chunk_text(text, source_name=f"WebURL ({url_clean[:30]}...)")
                if chunks:
                    new_chunks.extend(chunks)
                    indexed_signatures.add(sig)
                    messages.append(f"Parsed Web URL ({len(chunks)} chunks)")
                else:
                    messages.append(f"Web URL produced 0 text chunks.")
            else:
                messages.append(f"Web URL Error: {text}")

    if new_chunks:
        added = vector_store.add_chunks(new_chunks)
        return added, messages
    return 0, messages


# API Endpoints
@app.get("/api/status")
async def get_status():
    ollama_online = check_ollama_online(OLLAMA_BASE_URL)
    count = vector_store.count()
    return {
        "ollama_online": ollama_online,
        "local_model": LOCAL_MODEL_NAME,
        "vector_count": count
    }


@app.post("/api/index")
async def index_inputs(
    pdf_file: Optional[UploadFile] = File(None),
    img_file: Optional[UploadFile] = File(None),
    yt_url: Optional[str] = Form(""),
    web_url: Optional[str] = Form("")
):
    pdf_bytes = await pdf_file.read() if pdf_file else None
    pdf_name = pdf_file.filename if pdf_file else ""
    img_bytes = await img_file.read() if img_file else None
    img_name = img_file.filename if img_file else ""

    added, messages = auto_index_inputs(
        pdf_bytes=pdf_bytes,
        pdf_filename=pdf_name,
        img_bytes=img_bytes,
        img_filename=img_name,
        yt_url=yt_url or "",
        web_url=web_url or ""
    )

    return {
        "added": added,
        "messages": messages,
        "total_count": vector_store.count()
    }


@app.post("/api/clear")
async def clear_index():
    global indexed_signatures
    vector_store.clear_store()
    indexed_signatures.clear()
    return {"status": "cleared", "total_count": 0}


@app.post("/api/test_cloud")
async def test_cloud(
    cloud_base_url: str = Form(OPENROUTER_BASE_URL),
    cloud_model: str = Form(CLOUD_MODEL_NAME),
    cloud_api_key: str = Form(...)
):
    success, detail = test_cloud_connection(cloud_base_url, cloud_model, cloud_api_key)
    return {"success": success, "detail": detail}


@app.post("/api/audit")
async def run_audit(
    query: str = Form(...),
    cloud_base_url: str = Form(OPENROUTER_BASE_URL),
    cloud_model: str = Form(CLOUD_MODEL_NAME),
    cloud_api_key: str = Form(""),
    pdf_file: Optional[UploadFile] = File(None),
    img_file: Optional[UploadFile] = File(None),
    yt_url: Optional[str] = Form(""),
    web_url: Optional[str] = Form("")
):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Auto-index any uploaded files/links passed along with the query
    pdf_bytes = await pdf_file.read() if pdf_file else None
    pdf_name = pdf_file.filename if pdf_file else ""
    img_bytes = await img_file.read() if img_file else None
    img_name = img_file.filename if img_file else ""

    added, messages = auto_index_inputs(
        pdf_bytes=pdf_bytes,
        pdf_filename=pdf_name,
        img_bytes=img_bytes,
        img_filename=img_name,
        yt_url=yt_url or "",
        web_url=web_url or ""
    )

    # 2. Determine Time Sensitivity
    is_time_sensitive = is_time_sensitive_local(query, OLLAMA_BASE_URL, LOCAL_MODEL_NAME)

    # 3. Retrieve Document Context from ChromaDB
    doc_context = vector_store.search_chunks(query, top_k=DEFAULT_TOP_K)

    # 4. Retrieve Live Web Context if Time-Sensitive
    web_context = []
    if is_time_sensitive:
        web_context = fetch_live_web_context(query, max_results=2)

    # 5. LLM Synthesis
    response_text, provider_badge = generate_response(
        query=query,
        doc_context=doc_context,
        web_context=web_context,
        is_time_sensitive=is_time_sensitive,
        cloud_api_key=cloud_api_key,
        cloud_base_url=cloud_base_url,
        cloud_model=cloud_model,
        ollama_base_url=OLLAMA_BASE_URL,
        local_model=LOCAL_MODEL_NAME
    )

    return {
        "is_time_sensitive": is_time_sensitive,
        "doc_context": doc_context,
        "web_context": web_context,
        "synthesis": response_text,
        "provider_badge": provider_badge,
        "auto_indexed_added": added,
        "auto_indexed_messages": messages,
        "total_vector_count": vector_store.count()
    }


# Serve Embedded Single-Page Interactive Conversational Interface
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Radar — Conversational Intelligence</title>
    <!-- Google Fonts & Marked Markdown Parser -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

    <style>
        :root {
            --bg-primary: #0B0F19;
            --bg-surface: #1E293B;
            --bg-card: rgba(30, 41, 59, 0.75);
            --border-color: rgba(255, 255, 255, 0.1);
            --accent-primary: #3B82F6;
            --accent-purple: #8B5CF6;
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --success-bg: rgba(16, 185, 129, 0.15);
            --success-border: #10B981;
            --warning-bg: rgba(245, 158, 11, 0.15);
            --warning-border: #F59E0B;
            --danger-bg: rgba(239, 68, 68, 0.15);
            --danger-border: #EF4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 360px;
            background-color: #070B14;
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
            overflow-y: auto;
            gap: 1.2rem;
            flex-shrink: 0;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 1.3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .section-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 700;
            margin-top: 0.4rem;
        }

        .status-box {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-online {
            background: var(--success-bg);
            border: 1px solid var(--success-border);
            color: #34D399;
        }
        .status-offline {
            background: var(--warning-bg);
            border: 1px solid var(--warning-border);
            color: #FBBF24;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        input[type="text"], input[type="password"], select {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.6rem 0.8rem;
            border-radius: 6px;
            font-size: 0.88rem;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus, input[type="password"]:focus, select:focus {
            border-color: var(--accent-primary);
        }

        .file-upload-btn {
            background: rgba(255, 255, 255, 0.04);
            border: 1px dashed var(--border-color);
            padding: 0.75rem;
            border-radius: 6px;
            text-align: center;
            cursor: pointer;
            font-size: 0.82rem;
            color: var(--text-muted);
            transition: all 0.2s;
        }
        .file-upload-btn:hover {
            border-color: var(--accent-primary);
            color: var(--text-main);
            background: rgba(59, 130, 246, 0.1);
        }
        input[type="file"] {
            display: none;
        }

        .btn {
            padding: 0.7rem 1rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.88rem;
            cursor: pointer;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-purple) 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
        }
        .btn-primary:hover {
            opacity: 0.92;
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            color: var(--text-main);
        }
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        .btn-danger {
            background: var(--danger-bg);
            border: 1px solid var(--danger-border);
            color: #F87171;
        }
        .btn-danger:hover {
            background: rgba(239, 68, 68, 0.25);
        }

        /* Main Chat Layout */
        .main-chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            background: var(--bg-primary);
        }

        .chat-header {
            padding: 1.25rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(10px);
        }
        .chat-title {
            font-size: 1.2rem;
            font-weight: 700;
        }
        .chat-subtitle {
            font-size: 0.82rem;
            color: var(--text-muted);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        /* Message Bubbles */
        .message-row {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            max-width: 88%;
        }
        .message-row.user {
            align-self: flex-end;
        }
        .message-row.assistant {
            align-self: flex-start;
            width: 100%;
        }

        .user-bubble {
            background: linear-gradient(90deg, #2563EB 0%, #7C3AED 100%);
            color: white;
            padding: 0.9rem 1.2rem;
            border-radius: 18px 18px 4px 18px;
            font-size: 0.95rem;
            line-height: 1.5;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        }

        .assistant-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        /* Step Progress Banner */
        .progress-banner {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            padding: 0.85rem 1.2rem;
            font-size: 0.88rem;
            color: #60A5FA;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Accordion / Context Snippets */
        .context-split-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        .context-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.8rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .context-box-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .snippet-item {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 6px;
            padding: 0.65rem 0.8rem;
            font-size: 0.82rem;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }
        .snippet-source {
            font-size: 0.75rem;
            color: var(--accent-primary);
            font-weight: 600;
        }

        /* Markdown Formatting */
        .markdown-body {
            line-height: 1.6;
            color: #E2E8F0;
            font-size: 0.95rem;
        }
        .markdown-body h3 {
            color: #38BDF8;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        .markdown-body ul, .markdown-body ol {
            padding-left: 1.4rem;
            margin-bottom: 0.8rem;
        }
        .markdown-body code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(255, 255, 255, 0.1);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }

        /* Badges */
        .provider-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.8rem;
        }
        .badge-cloud {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid #10B981;
        }
        .badge-ollama {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid #F59E0B;
        }
        .badge-offline {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid #EF4444;
        }

        /* Chat Input Bar */
        .chat-input-area {
            padding: 1.25rem 2rem;
            border-top: 1px solid var(--border-color);
            background: rgba(15, 23, 42, 0.9);
            display: flex;
            gap: 0.8rem;
        }
        .chat-input-area input {
            flex: 1;
            font-size: 0.98rem;
            padding: 0.85rem 1.2rem;
            border-radius: 8px;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-left-color: var(--accent-primary);
            border-radius: 50%;
            width: 18px;
            height: 18px;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .toast-list {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }
        .toast-item {
            font-size: 0.8rem;
            color: #34D399;
        }
    </style>
</head>
<body>

    <!-- SIDEBAR -->
    <div class="sidebar">
        <div class="sidebar-header">
            <span>📡</span> Knowledge Radar
        </div>

        <!-- Local Ollama Status -->
        <div id="ollamaStatusBox" class="status-box status-offline">
            <span>🟡</span> Checking Ollama...
        </div>

        <div class="section-title">☁️ Cloud LLM Settings</div>

        <div class="form-group">
            <label for="providerSelect">Select LLM Provider</label>
            <select id="providerSelect" onchange="onProviderChange()">
                <option value="OpenRouter">OpenRouter (Free Tier - Recommended)</option>
                <option value="NVIDIA">NVIDIA NIM API</option>
                <option value="Custom">Custom OpenAI API</option>
            </select>
        </div>

        <div class="form-group">
            <label for="baseUrlInput">API Base URL</label>
            <input type="text" id="baseUrlInput" value="https://openrouter.ai/api/v1">
        </div>

        <div class="form-group">
            <label for="modelInput">Model Name</label>
            <input type="text" id="modelInput" value="meta-llama/llama-3.3-70b-instruct">
        </div>

        <div class="form-group">
            <label for="apiKeyInput">Cloud API Key (Instant Speed)</label>
            <input type="password" id="apiKeyInput" placeholder="sk-or-v1-...">
            <button class="btn btn-secondary" style="font-size:0.78rem; padding:0.4rem; margin-top:0.3rem;" onclick="testCloudConnection()">
                🔌 Test Cloud Connection
            </button>
        </div>

        <div class="section-title">📂 Multi-Modal Vault</div>

        <div class="form-group">
            <label>📄 Upload PDF Document</label>
            <div class="file-upload-btn" onclick="document.getElementById('pdfInput').click()">
                <span id="pdfLabel">Choose PDF File...</span>
            </div>
            <input type="file" id="pdfInput" accept=".pdf" onchange="updateFileLabel('pdfInput', 'pdfLabel')">
        </div>

        <div class="form-group">
            <label>🖼️ Upload Image (OCR)</label>
            <div class="file-upload-btn" onclick="document.getElementById('imgInput').click()">
                <span id="imgLabel">Choose Image File...</span>
            </div>
            <input type="file" id="imgInput" accept="image/*" onchange="updateFileLabel('imgInput', 'imgLabel')">
        </div>

        <div class="form-group">
            <label for="ytInput">🎥 YouTube Video Link</label>
            <input type="text" id="ytInput" placeholder="https://youtube.com/watch?v=...">
        </div>

        <div class="form-group">
            <label for="webInput">🌐 Web Page URL</label>
            <input type="text" id="webInput" placeholder="https://example.com/article">
        </div>

        <button class="btn btn-primary" id="btnIndex" onclick="indexVaultInputs()">
            <span>⚡</span> Process & Index Inputs
        </button>

        <div id="indexToasts" class="toast-list"></div>

        <div style="border-top: 1px solid var(--border-color); margin-top: 0.5rem; padding-top: 1rem; display:flex; flex-direction:column; gap:0.6rem;">
            <div style="font-size: 0.85rem; color: var(--text-muted);">
                📊 Active Vector Chunks: <strong id="vectorCountText" style="color:var(--text-main);">0</strong>
            </div>
            <button class="btn btn-secondary" style="width: 100%; font-size:0.8rem;" onclick="clearChatThread()">
                💬 Clear Chat Thread
            </button>
            <button class="btn btn-danger" style="width: 100%; font-size:0.8rem;" onclick="clearVectorStore()">
                🗑️ Clear Vector Index
            </button>
        </div>
    </div>

    <!-- MAIN CHAT CONTAINER -->
    <div class="main-chat-container">
        <div class="chat-header">
            <div>
                <div class="chat-title">📡 Knowledge Radar Intelligence Console</div>
                <div class="chat-subtitle">Conversational Multi-Modal Intelligence with Dual-Context Auditing & Web Verification</div>
            </div>
        </div>

        <!-- Chat History List -->
        <div class="chat-messages" id="chatMessages">
            <div class="assistant-card" id="welcomeMessage">
                <div style="font-size:1.1rem; font-weight:700; color:#38BDF8;">👋 Welcome to Knowledge Radar!</div>
                <div class="markdown-body">
                    Upload a <strong>PDF document</strong>, <strong>Image</strong>, <strong>YouTube URL</strong>, or <strong>Web page link</strong> in the sidebar, or ask a question directly.<br><br>
                    <strong>Features:</strong>
                    <ul>
                        <li><strong>⚡ Real-Time Transparency:</strong> Step-by-step progress status for every query.</li>
                        <li><strong>📄 Multi-Modal Context:</strong> Automatic ChromaDB vector indexing.</li>
                        <li><strong>🌐 Live Web Verification:</strong> DuckDuckGo search cross-checks time-sensitive prompts.</li>
                        <li><strong>💬 Conversational Memory:</strong> Chat continuously until your query is fully fulfilled.</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Live Step Progress Indicator -->
        <div id="progressBanner" style="display:none; padding:0 2rem 0.8rem 2rem;">
            <div class="progress-banner">
                <span class="spinner"></span>
                <span id="progressText">⚡ Processing query...</span>
            </div>
        </div>

        <!-- Chat Input Area -->
        <div class="chat-input-area">
            <input type="text" id="userQueryInput" placeholder="Ask a question across your documents or live web facts..." onkeypress="handleKeyPress(event)">
            <button class="btn btn-primary" id="btnSend" onclick="sendChatMessage()">
                <span id="btnIcon">⚡</span> Send
            </button>
        </div>
    </div>

    <script>
        const OPENROUTER_BASE = "https://openrouter.ai/api/v1";
        const OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free";
        const NVIDIA_BASE = "https://integrate.api.nvidia.com/v1";
        const NVIDIA_MODEL = "meta/llama-3.3-70b-instruct";

        window.onload = function() {
            fetchStatus();
            setInterval(fetchStatus, 8000);
        };

        async function testCloudConnection() {
            const key = document.getElementById('apiKeyInput').value.trim();
            const toasts = document.getElementById('indexToasts');
            if (!key) {
                toasts.innerHTML = '<div class="toast-item" style="color:#F87171;">⚠️ Please enter an API Key to test.</div>';
                return;
            }

            toasts.innerHTML = '<div class="toast-item" style="color:var(--accent-primary);">🔌 Testing Cloud API Key connection...</div>';

            const formData = new FormData();
            formData.append('cloud_base_url', document.getElementById('baseUrlInput').value);
            formData.append('cloud_model', document.getElementById('modelInput').value);
            formData.append('cloud_api_key', key);

            try {
                const res = await fetch('/api/test_cloud', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    toasts.innerHTML = `<div class="toast-item" style="color:#34D399;">🟢 ${escapeHtml(data.detail)}</div>`;
                } else {
                    toasts.innerHTML = `<div class="toast-item" style="color:#F87171;">🔴 ${escapeHtml(data.detail)}</div>`;
                }
            } catch (e) {
                toasts.innerHTML = `<div class="toast-item" style="color:#F87171;">🔴 Test Error: ${escapeHtml(e.message)}</div>`;
            }
        }

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendChatMessage();
        }

        function onProviderChange() {
            const val = document.getElementById('providerSelect').value;
            const baseUrlInput = document.getElementById('baseUrlInput');
            const modelInput = document.getElementById('modelInput');

            if (val === "OpenRouter") {
                baseUrlInput.value = OPENROUTER_BASE;
                modelInput.value = OPENROUTER_MODEL;
            } else if (val === "NVIDIA") {
                baseUrlInput.value = NVIDIA_BASE;
                modelInput.value = NVIDIA_MODEL;
            } else {
                baseUrlInput.value = "https://api.openai.com/v1";
                modelInput.value = "gpt-3.5-turbo";
            }
        }

        function updateFileLabel(inputId, labelId) {
            const input = document.getElementById(inputId);
            const label = document.getElementById(labelId);
            if (input.files && input.files.length > 0) {
                label.innerText = "📄 " + input.files[0].name;
            }
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                const box = document.getElementById('ollamaStatusBox');
                if (data.ollama_online) {
                    box.className = "status-box status-online";
                    box.innerHTML = `<span>🟢</span> <strong>Local Ollama Online</strong> (${data.local_model})`;
                } else {
                    box.className = "status-box status-offline";
                    box.innerHTML = `<span>🟡</span> <strong>Ollama Offline</strong> (Use Cloud Key for fast speed)`;
                }

                document.getElementById('vectorCountText').innerText = data.vector_count;
            } catch (e) {
                console.error("Error fetching status:", e);
            }
        }

        async function indexVaultInputs() {
            const pdfInput = document.getElementById('pdfInput');
            const imgInput = document.getElementById('imgInput');
            const ytInput = document.getElementById('ytInput');
            const webInput = document.getElementById('webInput');
            const btnIndex = document.getElementById('btnIndex');

            const pdfFile = pdfInput.files[0];
            const imgFile = imgInput.files[0];
            const ytUrl = ytInput.value.trim();
            const webUrl = webInput.value.trim();

            const toasts = document.getElementById('indexToasts');

            if (!pdfFile && !imgFile && !ytUrl && !webUrl) {
                toasts.innerHTML = '<div class="toast-item" style="color:#F59E0B;">⚠️ Please upload a PDF/Image or enter a URL first.</div>';
                return;
            }

            const formData = new FormData();
            if (pdfFile) formData.append('pdf_file', pdfFile);
            if (imgFile) formData.append('img_file', imgFile);
            if (ytUrl) formData.append('yt_url', ytUrl);
            if (webUrl) formData.append('web_url', webUrl);

            btnIndex.disabled = true;
            btnIndex.innerHTML = '<span class="spinner"></span> Indexing...';
            toasts.innerHTML = '<div class="toast-item" style="color:var(--accent-primary);">⚡ Extracting text & generating vector embeddings...</div>';

            try {
                const res = await fetch('/api/index', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                toasts.innerHTML = '';
                if (data.added > 0) {
                    toasts.innerHTML += `<div class="toast-item">✅ Indexed ${data.added} new chunks into ChromaDB!</div>`;
                }
                (data.messages || []).forEach(m => {
                    toasts.innerHTML += `<div class="toast-item">✓ ${m}</div>`;
                });
                if (data.added === 0 && (!data.messages || data.messages.length === 0)) {
                    toasts.innerHTML = '<div class="toast-item" style="color:var(--text-muted);">ℹ️ Inputs are already indexed in ChromaDB.</div>';
                }

                document.getElementById('vectorCountText').innerText = data.total_count;
            } catch (e) {
                toasts.innerHTML = `<div class="toast-item" style="color:#F87171;">🔴 Indexing Error: ${e.message}</div>`;
            } finally {
                btnIndex.disabled = false;
                btnIndex.innerHTML = '<span>⚡</span> Process & Index Inputs';
            }
        }

        async function clearVectorStore() {
            if (!confirm("Are you sure you want to clear the local ChromaDB index?")) return;
            try {
                const res = await fetch('/api/clear', { method: 'POST' });
                const data = await res.json();
                document.getElementById('vectorCountText').innerText = data.total_count;
                document.getElementById('indexToasts').innerHTML = '<div class="toast-item">Cleared ChromaDB index.</div>';
            } catch (e) {
                alert("Error clearing vector store: " + e.message);
            }
        }

        function clearChatThread() {
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = `
                <div class="assistant-card">
                    <div style="font-weight:700; color:#38BDF8;">💬 Chat thread cleared. Start a new conversation below!</div>
                </div>
            `;
        }

        function setProgressStep(msg) {
            const banner = document.getElementById('progressBanner');
            const text = document.getElementById('progressText');
            if (msg) {
                banner.style.display = "block";
                text.innerText = msg;
            } else {
                banner.style.display = "none";
            }
        }

        async function sendChatMessage() {
            const input = document.getElementById('userQueryInput');
            const query = input.value.trim();
            if (!query) return;

            // Render User Bubble
            const chatMessages = document.getElementById('chatMessages');
            const userRow = document.createElement('div');
            userRow.className = "message-row user";
            userRow.innerHTML = `<div class="user-bubble">${escapeHtml(query)}</div>`;
            chatMessages.appendChild(userRow);

            input.value = "";
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Disable Send Button
            const btn = document.getElementById('btnSend');
            const btnIcon = document.getElementById('btnIcon');
            btn.disabled = true;
            btnIcon.className = "spinner";
            btnIcon.innerText = "";

            // Show Progress Steps
            setProgressStep("⚡ Step 1: Processing inputs & checking ChromaDB vector store...");

            const formData = new FormData();
            formData.append('query', query);
            formData.append('cloud_base_url', document.getElementById('baseUrlInput').value);
            formData.append('cloud_model', document.getElementById('modelInput').value);
            formData.append('cloud_api_key', document.getElementById('apiKeyInput').value);

            const pdfInput = document.getElementById('pdfInput');
            const imgInput = document.getElementById('imgInput');
            if (pdfInput.files[0]) formData.append('pdf_file', pdfInput.files[0]);
            if (imgInput.files[0]) formData.append('img_file', imgInput.files[0]);
            formData.append('yt_url', document.getElementById('ytInput').value.trim());
            formData.append('web_url', document.getElementById('webInput').value.trim());

            try {
                setTimeout(() => setProgressStep("🌐 Step 2: Probing query time-sensitivity & searching web..."), 1200);
                setTimeout(() => setProgressStep("🤖 Step 3: Performing Dual-Context Audit via LLM Engine..."), 2500);

                const res = await fetch('/api/audit', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                // Update active vector count
                if (data.total_vector_count !== undefined) {
                    document.getElementById('vectorCountText').innerText = data.total_vector_count;
                }

                // Render Assistant Card
                const assistRow = document.createElement('div');
                assistRow.className = "message-row assistant";

                const docSnippets = (data.doc_context || []).map((s, i) => `
                    <div class="snippet-item">
                        <div class="snippet-source">Snippet #${i+1} — ${s.source}</div>
                        <div>${escapeHtml(s.text)}</div>
                    </div>
                `).join('');

                const webSnippets = (data.web_context || []).map(w => `
                    <div class="snippet-item">
                        <div class="snippet-source">🌐 ${escapeHtml(w.title)}</div>
                        <div>${escapeHtml(w.snippet)}</div>
                        <a href="${w.url}" target="_blank" style="color:var(--accent-primary); font-size:0.75rem;">${w.url}</a>
                    </div>
                `).join('');

                const badgeText = data.provider_badge || "Provider Ready";
                let badgeClass = "badge-offline";
                if (badgeText.includes("NVIDIA") || badgeText.includes("Cloud")) badgeClass = "badge-cloud";
                else if (badgeText.includes("Ollama")) badgeClass = "badge-ollama";

                const parsedMarkdown = (typeof marked !== 'undefined') ? marked.parse(data.synthesis || "") : data.synthesis;

                assistRow.innerHTML = `
                    <div class="assistant-card">
                        <div style="display:flex; align-items:center; justify-space-between;">
                            <div class="provider-badge ${badgeClass}">🏷️ ${badgeText}</div>
                            <span style="font-size:0.8rem; color:var(--text-muted);">Time Sensitive: ${data.is_time_sensitive ? "⚡ Yes" : "No"}</span>
                        </div>

                        <div class="context-split-grid">
                            <div class="context-box">
                                <div class="context-box-title">📄 Multi-Modal Context (${(data.doc_context || []).length})</div>
                                ${docSnippets || '<div style="font-size:0.8rem; color:var(--text-muted);">No document context found in ChromaDB.</div>'}
                            </div>
                            <div class="context-box-title" style="display:flex; flex-direction:column; gap:0.5rem; width:100%;">
                                <div class="context-box" style="width:100%;">
                                    <div class="context-box-title">🌐 Live Web Search (${(data.web_context || []).length})</div>
                                    ${webSnippets || '<div style="font-size:0.8rem; color:var(--text-muted);">Query bypassed live web search or returned no web results.</div>'}
                                </div>
                            </div>
                        </div>

                        <div class="markdown-body">
                            ${parsedMarkdown}
                        </div>
                    </div>
                `;

                chatMessages.appendChild(assistRow);
                chatMessages.scrollTop = chatMessages.scrollHeight;

            } catch (e) {
                const errRow = document.createElement('div');
                errRow.className = "message-row assistant";
                errRow.innerHTML = `<div class="assistant-card" style="border-color:#EF4444;"><div style="color:#F87171;">🚨 Error executing audit: ${e.message}</div></div>`;
                chatMessages.appendChild(errRow);
            } finally {
                setProgressStep(null);
                btn.disabled = false;
                btnIcon.className = "";
                btnIcon.innerText = "⚡";
            }
        }

        function escapeHtml(text) {
            if (!text) return "";
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    print("📡 Launching Knowledge Radar Conversational Console at http://127.0.0.1:8000...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
