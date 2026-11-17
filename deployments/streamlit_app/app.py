"""
Knowledge Radar — Streamlit Cloud Native Deployment.

Conversational Multi-Modal Intelligence Platform providing multi-source
document ingestion, local vector retrieval, and hybrid LLM claim auditing.
"""

import io
import os
import sys
import time
import uuid
import streamlit as st

# Ensure repository root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    LOCAL_MODEL_NAME, CLOUD_MODEL_NAME, OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL, NVIDIA_NIM_BASE_URL, GEMINI_BASE_URL,
    GEMINI_MODELS, DEFAULT_TOP_K
)
from modules.parser import parse_pdf, parse_youtube, parse_url, parse_image, chunk_sentences
from modules.vector_store import KnowledgeVectorStore
from modules.web_search import fetch_live_web_context
from modules.llm_router import check_ollama_online, is_time_sensitive_local, generate_response, test_cloud_connection
from modules.session_manager import SessionManager
from modules.turn_evaluator import evaluate_turn


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Knowledge Radar — Multi-Modal Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Metric & Ribbon Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.72rem;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
    }

    /* Badges */
    .badge-gemini {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-cloud {
        display: inline-block;
        background: rgba(168, 85, 247, 0.12);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-grounded {
        color: #34D399;
        font-weight: 600;
    }
    .badge-unverified {
        color: #F87171;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = KnowledgeVectorStore()

if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager(max_sessions=50, session_ttl_seconds=7200)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_signatures" not in st.session_state:
    st.session_state.indexed_signatures = set()


# -----------------------------------------------------------------------------
# Ingestion Helper Function
# -----------------------------------------------------------------------------
def ingest_inputs(pdf_files, img_files, yt_url: str, web_url: str):
    new_chunks = []
    messages = []

    # 1. PDFs
    if pdf_files:
        for pdf in pdf_files:
            file_bytes = pdf.getvalue()
            sig = f"pdf_{pdf.name}_{len(file_bytes)}"
            if sig not in st.session_state.indexed_signatures:
                text = parse_pdf(io.BytesIO(file_bytes))
                if text and not text.startswith("Error"):
                    chunks = chunk_sentences(text, source_name=f"PDF: {pdf.name}")
                    if chunks:
                        new_chunks.extend(chunks)
                        st.session_state.indexed_signatures.add(sig)
                        messages.append(f"Indexed PDF '{pdf.name}' ({len(chunks)} chunks)")
                else:
                    messages.append(f"PDF Notice: {text}")

    # 2. Images (OCR / Vision)
    if img_files:
        for img in img_files:
            file_bytes = img.getvalue()
            sig = f"img_{img.name}_{len(file_bytes)}"
            if sig not in st.session_state.indexed_signatures:
                text = parse_image(io.BytesIO(file_bytes))
                if text and not text.startswith("Error"):
                    chunks = chunk_sentences(text, source_name=f"Image: {img.name}")
                    if chunks:
                        new_chunks.extend(chunks)
                        st.session_state.indexed_signatures.add(sig)
                        messages.append(f"Indexed Image '{img.name}' ({len(chunks)} chunks)")
                else:
                    messages.append(f"Image Notice: {text}")

    # 3. YouTube URL
    if yt_url and yt_url.strip():
        clean_yt = yt_url.strip()
        sig = f"yt_{clean_yt}"
        if sig not in st.session_state.indexed_signatures:
            text = parse_youtube(clean_yt)
            if text and not text.startswith("Error"):
                chunks = chunk_sentences(text, source_name=f"YouTube ({clean_yt[:28]}...)")
                if chunks:
                    new_chunks.extend(chunks)
                    st.session_state.indexed_signatures.add(sig)
                    messages.append(f"Indexed YouTube transcript ({len(chunks)} chunks)")
            else:
                messages.append(f"YouTube Notice: {text}")

    # 4. Webpage URL
    if web_url and web_url.strip():
        clean_web = web_url.strip()
        sig = f"web_{clean_web}"
        if sig not in st.session_state.indexed_signatures:
            text = parse_url(clean_web)
            if text and not text.startswith("Error"):
                chunks = chunk_sentences(text, source_name=f"Web: {clean_web[:28]}...")
                if chunks:
                    new_chunks.extend(chunks)
                    st.session_state.indexed_signatures.add(sig)
                    messages.append(f"Indexed Web page ({len(chunks)} chunks)")
            else:
                messages.append(f"Web Notice: {text}")

    if new_chunks:
        added = st.session_state.vector_store.add_chunks(new_chunks)
        return added, messages
    return 0, messages


# -----------------------------------------------------------------------------
# Sidebar: Document Vault & Provider Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📡 Knowledge Radar")
    st.caption("Conversational Multi-Modal Intelligence Platform")
    st.markdown("---")

    sidebar_tab = st.radio("Navigation", ["📁 Document Vault", "⚙️ Model Settings"], horizontal=True)

    if sidebar_tab == "📁 Document Vault":
        st.subheader("Ingest Sources")

        pdf_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
        img_files = st.file_uploader("Upload Images (OCR / Charts)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        yt_url_input = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        web_url_input = st.text_input("Web Article URL", placeholder="https://en.wikipedia.org/wiki/...")

        col_ingest, col_clear = st.columns([2, 1])
        with col_ingest:
            if st.button("⚡ Ingest & Index Input", use_container_width=True, type="primary"):
                with st.spinner("Ingesting & generating vector embeddings..."):
                    added, msgs = ingest_inputs(pdf_files, img_files, yt_url_input, web_url_input)
                    if added > 0:
                        st.success(f"Indexed {added} new chunks into Vault!")
                    for m in msgs:
                        st.info(m)
                    if added == 0 and not msgs:
                        st.caption("No new inputs to index.")
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.vector_store.chunks = []
                st.session_state.indexed_signatures.clear()
                st.success("Cleared Vault.")

        st.markdown("---")
        st.subheader("Vault Inventory")
        sources_summary = st.session_state.vector_store.get_sources_summary()
        total_chunks = st.session_state.vector_store.count()
        st.metric("Total Indexed Chunks", total_chunks)

        if sources_summary:
            for src, count in sources_summary.items():
                icon = "📄"
                if "YouTube" in src:
                    icon = "🎥"
                elif "Image" in src:
                    icon = "🖼️"
                elif "Web" in src:
                    icon = "🌐"
                st.markdown(f"**{icon} {src}**: `{count} chunks`")
        else:
            st.caption("Vault is empty. Upload documents or links above.")

    else:
        st.subheader("Inference Engine")
        provider = st.selectbox("Provider", ["Google Gemini", "OpenRouter", "NVIDIA NIM", "Local Ollama"])

        if provider == "Google Gemini":
            gemini_mode = st.radio(
                "Model Selection Mode",
                ["Preset Tiers", "Custom Model Name"],
                horizontal=True
            )
            if gemini_mode == "Preset Tiers":
                gemini_tier = st.select_slider(
                    "Gemini Speed vs. Reasoning Tier",
                    options=["low", "medium", "high"],
                    value="medium",
                    format_func=lambda x: {
                        "low": "⚡ Low (2.0 Flash-Lite)",
                        "medium": "⚖️ Medium (2.0 Flash)",
                        "high": "🧠 High (1.5 Pro)"
                    }[x]
                )
                model_name = GEMINI_MODELS[gemini_tier]
            else:
                gemini_tier = "custom"
                model_name = st.text_input(
                    "Gemini Model ID",
                    value="gemini-2.0-flash",
                    help="E.g. gemini-2.0-flash, gemini-1.5-pro, gemini-2.0-flash-lite, gemini-2.5-pro"
                ).strip()
            base_url = GEMINI_BASE_URL
            api_key = st.text_input("Gemini API Key (Google AI Studio)", type="password", placeholder="AIzaSy...")

        elif provider == "OpenRouter":
            gemini_tier = "medium"
            model_name = st.text_input(
                "Model ID",
                value="meta-llama/llama-3.3-70b-instruct:free",
                help="Any valid OpenRouter model slug (e.g. deepseek/deepseek-r1, anthropic/claude-3.5-sonnet)"
            ).strip()
            base_url = OPENROUTER_BASE_URL
            api_key = st.text_input("OpenRouter API Key", type="password", placeholder="sk-or-v1-...")

        elif provider == "NVIDIA NIM":
            gemini_tier = "medium"
            model_name = st.text_input(
                "Model ID",
                value="meta/llama-3.3-70b-instruct",
                help="NVIDIA NIM model ID (e.g. meta/llama-3.3-70b-instruct, deepseek-ai/deepseek-r1)"
            ).strip()
            base_url = NVIDIA_NIM_BASE_URL
            api_key = st.text_input("NVIDIA API Key", type="password", placeholder="nvapi-...")

        else:
            gemini_tier = "medium"
            model_name = st.text_input("Ollama Model", value=LOCAL_MODEL_NAME).strip()
            base_url = OLLAMA_BASE_URL
            api_key = ""

        st.session_state.provider = provider
        st.session_state.model_name = model_name
        st.session_state.base_url = base_url
        st.session_state.api_key = api_key
        st.session_state.gemini_tier = gemini_tier

        # Test Connection button for Cloud & Local Ollama
        if provider == "Local Ollama":
            if st.button("🔌 Test Ollama Connection", use_container_width=True):
                with st.spinner(f"Pinging Ollama at {base_url}..."):
                    is_online = check_ollama_online(base_url)
                    if is_online:
                        st.success(f"Ollama is reachable at {base_url}! Model: {model_name}")
                    else:
                        st.error(f"Cannot reach Ollama at {base_url}. Ensure 'ollama serve' is running.")
        else:
            if st.button("🔌 Test Model Connection", use_container_width=True):
                if not api_key or not api_key.strip():
                    st.warning("Please enter your API Key to test the connection.")
                elif not model_name:
                    st.warning("Please specify a valid model name.")
                else:
                    with st.spinner(f"Testing connection with model '{model_name}'..."):
                        is_ok, detail = test_cloud_connection(base_url, model_name, api_key)
                        if is_ok:
                            st.success(detail)
                        else:
                            st.error(detail)


# -----------------------------------------------------------------------------
# Top Header & Multi-Turn Session Ribbon
# -----------------------------------------------------------------------------
session_summary = st.session_state.session_manager.get_session_summary(st.session_state.session_id)
turn_count = session_summary.get("turn_count", 0)
cum_tokens = session_summary.get("cumulative_tokens", 0)
avg_faithfulness = session_summary.get("avg_faithfulness", 1.0)
avg_faith_pct = round(avg_faithfulness * 100)

ribbon_col1, ribbon_col2, ribbon_col3, ribbon_col4, ribbon_col5 = st.columns([1.5, 1.5, 1.5, 2, 1.5])
with ribbon_col1:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>#{st.session_state.session_id[:8]}</div><div class='metric-lbl'>Session ID</div></div>", unsafe_allow_html=True)
with ribbon_col2:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{turn_count}</div><div class='metric-lbl'>Total Turns</div></div>", unsafe_allow_html=True)
with ribbon_col3:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{cum_tokens:,}</div><div class='metric-lbl'>Cumulative Tokens</div></div>", unsafe_allow_html=True)
with ribbon_col4:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{avg_faith_pct}%</div><div class='metric-lbl'>Avg Groundedness</div></div>", unsafe_allow_html=True)
with ribbon_col5:
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.session_manager.reset_session(st.session_state.session_id)
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.rerun()

st.markdown("---")


# -----------------------------------------------------------------------------
# Render Conversational Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            # Header Meta Strip
            badge_html = f"<span class='badge-gemini'>{msg.get('provider_badge', 'AI Assistant')}</span>"
            tokens_str = f"{msg.get('tokens', 0):,} tok"
            latency_str = f"{msg.get('latency_ms', 0):.0f} ms"
            faith_pct = round(msg.get('faithfulness', 1.0) * 100)

            st.markdown(
                f"{badge_html} &nbsp; <b>Turn #{msg.get('turn_index', 1)}</b> • ⏱️ {latency_str} • 🪙 {tokens_str} • 🎯 <b>{faith_pct}% Grounded</b>",
                unsafe_allow_html=True
            )

            # Answer Markdown
            st.markdown(msg["content"])

            # Collapsible Audit Drawer
            claims_audit = msg.get("claims_audit", [])
            doc_context = msg.get("doc_context", [])
            web_context = msg.get("web_context", [])
            latencies = msg.get("latencies", {})

            with st.expander(f"🔍 Inspect Evidence & Claim Audit ({len(doc_context)} chunks, {len(claims_audit)} claims)"):
                tab_claims, tab_docs, tab_web, tab_telemetry = st.tabs([
                    f"Claim Audit ({len(claims_audit)})",
                    f"Document Chunks ({len(doc_context)})",
                    f"Web Verification ({len(web_context)})",
                    "Latency & Math"
                ])

                with tab_claims:
                    if claims_audit:
                        for idx, c in enumerate(claims_audit, start=1):
                            status_badge = "🟢 Grounded" if c["status"] == "GROUNDED" else ("🟡 Partial" if c["status"] == "PARTIAL" else "🔴 Unverified")
                            st.markdown(f"**#{idx} [{status_badge}]** `{c.get('match_score', 0):.0%}` — {c.get('claim', '')}")
                            if c.get("matched_passage"):
                                st.caption(f"↳ Evidence: \"{c['matched_passage']}\"")
                    else:
                        st.caption("No discrete factual assertions detected.")

                with tab_docs:
                    if doc_context:
                        for d in doc_context:
                            st.markdown(f"**{d.get('source', 'Vault')}** (Score: `{d.get('similarity', 0):.3f}`)")
                            st.info(d.get("text", ""))
                    else:
                        st.caption("No internal documents retrieved.")

                with tab_web:
                    if web_context:
                        for w in web_context:
                            st.markdown(f"**[{w.get('title', 'Web Article')}]({w.get('url', '#')})**")
                            st.caption(w.get("snippet", ""))
                    else:
                        st.caption("No external web verification was required or retrieved.")

                with tab_telemetry:
                    col_t1, col_t2, col_t3 = st.columns(3)
                    with col_t1:
                        st.metric("Retrieval Latency", f"{latencies.get('retrieval', 0):.1f} ms")
                    with col_t2:
                        st.metric("Web Search Latency", f"{latencies.get('web', 0):.1f} ms")
                    with col_t3:
                        st.metric("Synthesis Latency", f"{latencies.get('synthesis', 0):.1f} ms")


# -----------------------------------------------------------------------------
# Chat Input & Multi-Source Synthesis Turn
# -----------------------------------------------------------------------------
prompt = st.chat_input("Ask a question across your documents and live web...")

if prompt:
    # 1. Render User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Execute Audit Turn
    with st.chat_message("assistant"):
        with st.status("📡 Knowledge Radar Auditing...", expanded=True) as status:
            st.write("⚡ Step 1: Checking time-sensitivity and contextualizing memory...")
            session_state = st.session_state.session_manager.get_or_create(st.session_state.session_id)
            effective_query = st.session_state.session_manager.contextualize_query(prompt, st.session_state.session_id)

            is_time_sensitive = is_time_sensitive_local(effective_query, OLLAMA_BASE_URL, LOCAL_MODEL_NAME)

            # Web Search
            web_context = []
            web_start = time.time()
            if is_time_sensitive:
                st.write("🌐 Step 2: Query flagged as time-sensitive. Fetching live web verification context...")
                web_context = fetch_live_web_context(effective_query, max_results=2)
            else:
                st.write("🌐 Step 2: Time-sensitivity check: Internal vault focus.")
            web_latency_ms = round((time.time() - web_start) * 1000, 2)

            # Document Retrieval
            st.write("🔍 Step 3: Searching Document Vault across active sources...")
            retrieval_start = time.time()
            active_sources_count = len(st.session_state.vector_store.get_sources_summary())
            effective_top_k = max(DEFAULT_TOP_K, min(12, active_sources_count * 4))
            doc_context = st.session_state.vector_store.search_chunks(effective_query, top_k=effective_top_k)
            retrieval_latency_ms = round((time.time() - retrieval_start) * 1000, 2)

            # Cloud / LLM Synthesis
            st.write("🤖 Step 4: Generating grounded synthesis...")
            provider = getattr(st.session_state, "provider", "Google Gemini")
            base_url = getattr(st.session_state, "base_url", GEMINI_BASE_URL)
            model_name = getattr(st.session_state, "model_name", "gemini-2.0-flash")
            api_key = getattr(st.session_state, "api_key", "")
            chat_history = st.session_state.session_manager.get_recent_history(st.session_state.session_id, max_turns=3)

            response_text, provider_badge, telemetry = generate_response(
                query=prompt,
                doc_context=doc_context,
                web_context=web_context,
                is_time_sensitive=is_time_sensitive,
                cloud_api_key=api_key,
                cloud_base_url=base_url,
                cloud_model=model_name,
                ollama_base_url=OLLAMA_BASE_URL,
                local_model=LOCAL_MODEL_NAME,
                chat_history=chat_history
            )

            # Grounding & Turn Evaluation
            st.write("📊 Step 5: Auditing factual claims and groundedness score...")
            turn_eval = evaluate_turn(
                query=prompt,
                synthesis=response_text,
                doc_context=doc_context,
                web_context=web_context,
                telemetry=telemetry,
                retrieval_latency_ms=retrieval_latency_ms,
                web_latency_ms=web_latency_ms,
                session_history_faithfulness=session_state.running_faithfulness
            )

            # Record turn in session manager
            st.session_state.session_manager.add_turn(
                session_id=st.session_state.session_id,
                query=prompt,
                effective_query=effective_query,
                synthesis=response_text,
                metrics=turn_eval["metrics"]
            )

            status.update(label="✅ Audit Complete!", state="complete", expanded=False)

        # Display Response
        total_tokens = telemetry.get("total_tokens", 0)
        total_latency = turn_eval["metrics"]["turn"].get("total_latency_ms", 0)
        faith_val = turn_eval["metrics"]["turn"].get("faithfulness", 1.0)
        faith_pct = round(faith_val * 100)

        badge_html = f"<span class='badge-gemini'>{provider_badge}</span>"
        st.markdown(
            f"{badge_html} &nbsp; <b>Turn #{len(session_state.turns)}</b> • ⏱️ {total_latency:.0f} ms • 🪙 {total_tokens:,} tok • 🎯 <b>{faith_pct}% Grounded</b>",
            unsafe_allow_html=True
        )
        st.markdown(response_text)

        # Append to session messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "provider_badge": provider_badge,
            "turn_index": len(session_state.turns),
            "tokens": total_tokens,
            "latency_ms": total_latency,
            "faithfulness": faith_val,
            "claims_audit": turn_eval["claims_audit"],
            "doc_context": doc_context,
            "web_context": web_context,
            "latencies": {
                "retrieval": retrieval_latency_ms,
                "web": web_latency_ms,
                "synthesis": telemetry.get("synthesis_latency_ms", 0.0)
            }
        })
        st.rerun()
