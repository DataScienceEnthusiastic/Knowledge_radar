"""
Knowledge Radar — Conversational Multi-Modal Intelligence Platform.

FastAPI application providing multi-modal ingestion (PDFs, YouTube transcripts,
web URLs, and OCR images), local vector retrieval, and hybrid LLM auditing.
"""

import os
import io
import time
import tempfile
import uvicorn
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import (
    LOCAL_MODEL_NAME, CLOUD_MODEL_NAME, OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL, NVIDIA_NIM_BASE_URL, DEFAULT_TOP_K
)
from modules.parser import parse_pdf, parse_youtube, parse_url, parse_image, chunk_text, chunk_sentences
from modules.vector_store import KnowledgeVectorStore
from modules.web_search import fetch_live_web_context
from modules.llm_router import check_ollama_online, is_time_sensitive_local, generate_response, test_cloud_connection
from modules.session_manager import SessionManager
from modules.turn_evaluator import evaluate_turn

app = FastAPI(title="Knowledge Radar API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton vector store and conversational session manager
vector_store = KnowledgeVectorStore()
indexed_signatures = set()
session_manager = SessionManager(max_sessions=100, session_ttl_seconds=3600)


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
                chunks = chunk_sentences(text, source_name=f"PDF: {pdf_filename}")
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
                chunks = chunk_sentences(text, source_name=f"Image: {img_filename}")
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
                chunks = chunk_sentences(text, source_name=f"YouTube ({url_clean[:30]}...)")
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
                chunks = chunk_sentences(text, source_name=f"WebURL ({url_clean[:30]}...)")
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
        "vector_count": count,
        "sources_summary": vector_store.get_sources_summary()
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
        "total_count": vector_store.count(),
        "sources_summary": vector_store.get_sources_summary()
    }


@app.post("/api/clear")
async def clear_index():
    global indexed_signatures
    vector_store.clear_store()
    indexed_signatures.clear()
    return {"status": "cleared", "total_count": 0, "sources_summary": {}}


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
    session_id: Optional[str] = Form(""),
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

    # 1. Manage Conversational Session Memory
    session_state = session_manager.get_or_create(session_id)
    active_session_id = session_state.session_id

    # 2. Auto-index any uploaded files/links passed along with the query
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

    # 3. Contextualize query with multi-turn conversational memory (resolves anaphora/pronouns)
    effective_query = session_manager.contextualize_query(query, active_session_id)

    # 4. Determine Time Sensitivity
    is_time_sensitive = is_time_sensitive_local(effective_query, OLLAMA_BASE_URL, LOCAL_MODEL_NAME)

    # 5. Retrieve Document Context from ChromaDB with latency timing
    t_ret_0 = time.perf_counter()
    active_sources_count = len(vector_store.get_sources_summary())
    effective_top_k = max(DEFAULT_TOP_K, min(12, active_sources_count * 4))
    doc_context = vector_store.search_chunks(effective_query, top_k=effective_top_k)
    retrieval_latency_ms = (time.perf_counter() - t_ret_0) * 1000.0

    # 6. Retrieve Live Web Context if Time-Sensitive with latency timing
    t_web_0 = time.perf_counter()
    web_context = []
    if is_time_sensitive:
        web_context = fetch_live_web_context(effective_query, max_results=2)
    web_latency_ms = (time.perf_counter() - t_web_0) * 1000.0

    # 7. Multi-Turn LLM Synthesis with conversational history
    chat_history = session_manager.get_recent_history(active_session_id, max_turns=3)
    response_text, provider_badge, telemetry = generate_response(
        query=query,
        doc_context=doc_context,
        web_context=web_context,
        is_time_sensitive=is_time_sensitive,
        cloud_api_key=cloud_api_key,
        cloud_base_url=cloud_base_url,
        cloud_model=cloud_model,
        ollama_base_url=OLLAMA_BASE_URL,
        local_model=LOCAL_MODEL_NAME,
        chat_history=chat_history
    )

    # 8. Live Turn Evaluation, Claim Grounding & Session Drift Tracking
    turn_eval = evaluate_turn(
        query=query,
        synthesis=response_text,
        doc_context=doc_context,
        web_context=web_context,
        telemetry=telemetry,
        retrieval_latency_ms=retrieval_latency_ms,
        web_latency_ms=web_latency_ms,
        session_history_faithfulness=session_state.running_faithfulness
    )

    # 9. Record turn in conversational session store
    session_manager.add_turn(
        session_id=active_session_id,
        query=query,
        effective_query=effective_query,
        synthesis=response_text,
        metrics=turn_eval["metrics"]
    )

    session_summary = session_manager.get_session_summary(active_session_id)

    return {
        "session_id": active_session_id,
        "turn_index": len(session_state.turns),
        "is_time_sensitive": is_time_sensitive,
        "effective_query": effective_query,
        "doc_context": doc_context,
        "web_context": web_context,
        "synthesis": response_text,
        "provider_badge": provider_badge,
        "auto_indexed_added": added,
        "auto_indexed_messages": messages,
        "total_vector_count": vector_store.count(),
        "sources_summary": vector_store.get_sources_summary(),
        "metrics": turn_eval["metrics"]["turn"],
        "drift": turn_eval["drift"],
        "claims_audit": turn_eval["claims_audit"],
        "session_summary": session_summary
    }


@app.post("/api/session/reset")
async def reset_session(session_id: str = Form(...)):
    cleared = session_manager.reset_session(session_id)
    return {
        "session_id": session_id,
        "cleared": cleared,
        "summary": session_manager.get_session_summary(session_id)
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    return session_manager.get_session_summary(session_id)


TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")

@app.get("/", response_class=FileResponse)
async def get_dashboard():
    if not os.path.exists(TEMPLATE_PATH):
        raise HTTPException(status_code=500, detail="Dashboard template not found.")
    return FileResponse(TEMPLATE_PATH, media_type="text/html")


if __name__ == "__main__":
    print("📡 Launching Knowledge Radar Conversational Console at http://127.0.0.1:8000...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
