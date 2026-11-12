"""
Hybrid LLM Router (Cloud Primary + Local Ollama Fallback).

Routes synthesis between Cloud LLMs (OpenRouter / NVIDIA NIM) and local Ollama,
handling rate limits, authorization failures, and offline states.
"""

import logging
import re
import time
import requests
from openai import OpenAI
from config import OLLAMA_BASE_URL, LOCAL_MODEL_NAME, CLOUD_MODEL_NAME, OPENROUTER_BASE_URL

logger = logging.getLogger("llm_router")

_OLLAMA_STATUS_CACHE = {"online": False, "checked_at": 0.0}

def check_ollama_online(ollama_base_url: str = OLLAMA_BASE_URL) -> bool:
    """Checks if the local Ollama server is running and reachable with a 5-second cache."""
    now = time.time()
    if now - _OLLAMA_STATUS_CACHE["checked_at"] < 5.0:
        return _OLLAMA_STATUS_CACHE["online"]

    try:
        host = ollama_base_url.rsplit("/v1", 1)[0]
        resp = requests.get(host, timeout=0.5)
        is_online = (resp.status_code == 200)
    except requests.RequestException:
        is_online = False

    _OLLAMA_STATUS_CACHE["online"] = is_online
    _OLLAMA_STATUS_CACHE["checked_at"] = now
    return is_online


def is_time_sensitive_local(query: str, ollama_base_url: str = OLLAMA_BASE_URL, local_model: str = LOCAL_MODEL_NAME) -> bool:
    """
    Checks if a query requires live web verification via local Ollama.
    Falls back to deterministic keyword matching if Ollama is unreachable.
    """
    if check_ollama_online(ollama_base_url):
        try:
            client = OpenAI(
                api_key="ollama",
                base_url=ollama_base_url,
                timeout=3.0
            )
            prompt = (
                "Determine if the user query requires live real-time web verification or recent current data (like prices, news, 2025/2026 facts, live status).\n"
                f"User Query: \"{query}\"\n"
                "Reply EXACTLY with 'YES' or 'NO' and nothing else."
            )
            res = client.chat.completions.create(
                model=local_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
                timeout=3.0
            )
            reply = res.choices[0].message.content.strip().upper()
            return "YES" in reply
        except Exception as e:
            logger.debug("Local Ollama time-sensitivity probe failed, falling back to keywords: %s", e)

    time_keywords = [
        "latest", "current", "today", "now", "recent", "2024", "2025", "2026",
        "version", "status", "price", "rate", "news", "trend", "market", "update",
        "real-time", "live", "competitor", "stock", "valuation"
    ]
    query_lower = query.lower()
    return any(re.search(rf"\b{kw}\b", query_lower) for kw in time_keywords)


def test_cloud_connection(base_url: str, model: str, api_key: str) -> tuple:
    """
    Tests connection and API key validity for Cloud LLMs (OpenRouter / NVIDIA NIM).
    """
    if not api_key or not api_key.strip():
        return False, "API Key is empty. Please enter your OpenRouter or NVIDIA NIM key."
        
    try:
        client = OpenAI(
            api_key=api_key.strip(),
            base_url=base_url.strip(),
            timeout=10.0
        )
        res = client.chat.completions.create(
            model=model.strip(),
            messages=[{"role": "user", "content": "Ping test. Reply with 'OK'."}],
            max_tokens=5,
            temperature=0.0,
            timeout=10.0
        )
        reply = res.choices[0].message.content
        return True, f"Connection Successful! Model '{model}' responded: '{reply.strip()}'"
    except Exception as e:
        return False, f"Cloud API Connection Error: {str(e)}"


def generate_response(
    query: str,
    doc_context: list,
    web_context: list,
    is_time_sensitive: bool,
    cloud_api_key: str = "",
    cloud_base_url: str = OPENROUTER_BASE_URL,
    cloud_model: str = CLOUD_MODEL_NAME,
    ollama_base_url: str = OLLAMA_BASE_URL,
    local_model: str = LOCAL_MODEL_NAME,
    chat_history: list = None
) -> tuple:
    """
    Performs Dual-Context Audit synthesis with multi-turn memory support.
    Tries Cloud LLM first; if cloud call fails or rate-limits (429), falls back to local Ollama.
    
    Returns:
        tuple: (response_markdown_text, provider_badge_name, telemetry_dict)
    """
    # 1. Format Context Inputs Grouped by Source
    doc_str = ""
    if doc_context:
        sources_grouped = {}
        for item in doc_context:
            src = item.get("source", "Document")
            sources_grouped.setdefault(src, []).append(item)

        for src, chunks in sources_grouped.items():
            doc_str += f"\n### SOURCE: {src} ({len(chunks)} passages):\n"
            for item in chunks:
                doc_str += f"- [Chunk {item['chunk_index']}]: {item['text']}\n"
    else:
        doc_str = "No relevant passages found in the uploaded context."

    web_str = ""
    if web_context:
        for item in web_context:
            web_str += f"- [{item['title']}] ({item['url']}): {item['snippet']}\n"
    else:
        web_str = "No web search executed or no web snippets retrieved."

    system_prompt = (
        "You are Knowledge Radar, an elite AI Intelligence Auditor. "
        "Your goal is to provide a comprehensive, transparent Dual-Context Audit based on:\n"
        "1. Uploaded Multi-Modal Context (PDFs, YouTube transcripts, Web URLs, Images)\n"
        "2. Live Web Verification (Real-time web search facts)\n\n"
        "Structuring Guidelines:\n"
        "- Format clean, executive Markdown text.\n"
        "- Split your analysis strictly into 3 sections:\n"
        "  ### 📄 Document Facts & Insights\n"
        "  ### 🌐 Live Web Verification & Audit\n"
        "  ### 🎯 Knowledge Radar Final Verdict\n"
        "- Multi-Source Synthesis: When multiple distinct sources (such as a PDF and a YouTube transcript) are provided, compare and correlate them directly in your analysis. Highlight how they relate, complement, or differ from each other.\n"
        "- Highlight any conflicts, updates, or omissions between uploaded documents and live web facts.\n"
        "- Uncertainty & Verification Discipline: Whenever stating facts, statistics, or dates NOT explicitly present in the provided context, or if uncertain, you MUST explicitly flag that statement with '[⚠️ Verification Required: reason]' so the user can verify it independently."
    )

    user_prompt = (
        f"USER QUERY: {query}\n\n"
        f"TIME-SENSITIVITY DETECTED: {'Yes' if is_time_sensitive else 'No'}\n\n"
        f"--- SOURCE 1: UPLOADED MULTI-MODAL CONTEXT ---\n{doc_str}\n\n"
        f"--- SOURCE 2: LIVE WEB SEARCH VERIFICATION ---\n{web_str}\n\n"
        "Please perform the Dual-Context Audit now."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    cloud_error = ""

    # 2. Primary Execution: Cloud LLM
    if cloud_api_key and cloud_api_key.strip():
        try:
            cloud_client = OpenAI(
                api_key=cloud_api_key.strip(),
                base_url=cloud_base_url.strip(),
                timeout=30.0
            )
            t0 = time.perf_counter()
            res = cloud_client.chat.completions.create(
                model=cloud_model.strip(),
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
                timeout=30.0
            )
            t1 = time.perf_counter()
            output = res.choices[0].message.content
            if "generativelanguage.googleapis.com" in cloud_base_url or "gemini" in cloud_model.lower():
                provider_label = f"Google Gemini ({cloud_model})"
            elif "nvidia" in cloud_base_url:
                provider_label = "NVIDIA Cloud"
            elif "openrouter" in cloud_base_url:
                provider_label = "Cloud LLM (OpenRouter)"
            else:
                provider_label = f"Cloud LLM ({cloud_model})"

            usage = getattr(res, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)
            if not total_tokens:
                prompt_tokens = len(user_prompt.split()) + len(system_prompt.split())
                completion_tokens = len(output.split())
                total_tokens = prompt_tokens + completion_tokens

            telemetry = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "synthesis_latency_ms": round((t1 - t0) * 1000, 2)
            }
            return output, f"Powered by {provider_label}", telemetry
        except Exception as e:
            cloud_error = str(e)
            logger.warning("Cloud LLM failed (%s). Attempting local Ollama fallback.", cloud_error)

    # 3. Secondary Execution: Fallback to Local Ollama
    if check_ollama_online(ollama_base_url):
        try:
            local_client = OpenAI(
                api_key="ollama",
                base_url=ollama_base_url,
                timeout=15.0
            )
            t0 = time.perf_counter()
            res = local_client.chat.completions.create(
                model=local_model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
                timeout=15.0
            )
            t1 = time.perf_counter()
            output = res.choices[0].message.content

            usage = getattr(res, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else len(user_prompt.split())
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else len(output.split())
            total_tokens = prompt_tokens + completion_tokens

            telemetry = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "synthesis_latency_ms": round((t1 - t0) * 1000, 2)
            }
            return output, "Powered by Local Ollama (Fallback Active)", telemetry
        except Exception as e:
            ollama_err = str(e)
            telemetry = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "synthesis_latency_ms": 0.0}
            if cloud_error:
                return (
                    f"🚨 **Cloud LLM Error**: `{cloud_error}`\n\n"
                    f"🚨 **Local Ollama Fallback Error**: `{ollama_err}`",
                    "API Execution Failed",
                    telemetry
                )
            return f"🚨 **Local Ollama Error**: `{ollama_err}`", "Local Ollama Error", telemetry

    # 4. If Cloud failed & Ollama offline:
    telemetry = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "synthesis_latency_ms": 0.0}
    if cloud_error:
        return (
            f"🚨 **Cloud LLM API Error**: `{cloud_error}`\n\n"
            "Please check your API key, Base URL, or Model Name in the sidebar.",
            "Cloud API Error",
            telemetry
        )

    return (
        "⚠️ **LLM Service Unavailable**:\n"
        "- Please enter a valid Cloud API Key (OpenRouter / NVIDIA NIM) in the sidebar **OR**\n"
        "- Ensure local Ollama is running (`ollama serve`) at `http://localhost:11434`.",
        "No Active LLM Provider",
        telemetry
    )
