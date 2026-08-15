"""
===============================================================================
🤖 MODULE: Hybrid LLM Router (Cloud Primary + Local Ollama Fallback)
===============================================================================
Uses standard OpenAI Python SDK to route requests between:
1. Local AI (Ollama at http://localhost:11434/v1) for time-sensitivity classification and fallback synthesis.
2. Cloud LLMs (NVIDIA NIM or OpenRouter) for heavy dual-context synthesis.
Handles rate limits (429), authorization errors, and offline states seamlessly.
===============================================================================
"""

import re
import requests
from openai import OpenAI
from config import OLLAMA_BASE_URL, LOCAL_MODEL_NAME, CLOUD_MODEL_NAME, OPENROUTER_BASE_URL


def check_ollama_online(ollama_base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Checks if the local Ollama server is running and reachable.
    
    Args:
        ollama_base_url (str): Base URL for Ollama's OpenAI-compatible endpoint.
        
    Returns:
        bool: True if reachable, False otherwise.
    """
    try:
        # Strip /v1 to ping root endpoint
        host = ollama_base_url.rsplit("/v1", 1)[0]
        resp = requests.get(host, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def is_time_sensitive_local(query: str, ollama_base_url: str = OLLAMA_BASE_URL, local_model: str = LOCAL_MODEL_NAME) -> bool:
    """
    Asks local Ollama if a query requires live web verification.
    Falls back to keyword matching if Ollama is unreachable.
    
    Args:
        query (str): User prompt.
        ollama_base_url (str): Ollama API endpoint.
        local_model (str): Ollama model tag (e.g., llama3.2:3b).
        
    Returns:
        bool: True if time-sensitive / needs live search.
    """
    if check_ollama_online(ollama_base_url):
        try:
            client = OpenAI(
                api_key="ollama",  # Dummy key required by SDK
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
        except Exception:
            pass

    # Keyword Fallback Rule
    time_keywords = [
        "latest", "current", "today", "now", "recent", "2024", "2025", "2026",
        "version", "status", "price", "rate", "news", "trend", "market", "update",
        "real-time", "live", "competitor", "stock", "valuation"
    ]
    query_lower = query.lower()
    return any(re.search(rf"\b{kw}\b", query_lower) for kw in time_keywords)


def generate_response(
    query: str,
    doc_context: list,
    web_context: list,
    is_time_sensitive: bool,
    cloud_api_key: str = "",
    cloud_base_url: str = OPENROUTER_BASE_URL,
    cloud_model: str = CLOUD_MODEL_NAME,
    ollama_base_url: str = OLLAMA_BASE_URL,
    local_model: str = LOCAL_MODEL_NAME
) -> tuple:
    """
    Performs Dual-Context Audit synthesis.
    Tries Cloud LLM first; if cloud call fails or rate-limits (429), falls back to local Ollama.
    
    Returns:
        tuple: (response_markdown_text, provider_badge_name)
    """
    # 1. Format Context Inputs
    doc_str = ""
    if doc_context:
        for item in doc_context:
            doc_str += f"- [{item['source']} | Chunk {item['chunk_index']}]: {item['text']}\n"
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
        "- Highlight any conflicts, updates, or omissions between uploaded documents and live web facts."
    )

    user_prompt = (
        f"USER QUERY: {query}\n\n"
        f"TIME-SENSITIVITY DETECTED: {'Yes' if is_time_sensitive else 'No'}\n\n"
        f"--- SOURCE 1: UPLOADED MULTI-MODAL CONTEXT ---\n{doc_str}\n\n"
        f"--- SOURCE 2: LIVE WEB SEARCH VERIFICATION ---\n{web_str}\n\n"
        "Please perform the Dual-Context Audit now."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def test_cloud_connection(base_url: str, model: str, api_key: str) -> tuple:
    """
    Tests connection and API key validity for Cloud LLMs (OpenRouter / NVIDIA NIM).
    
    Returns:
        tuple: (success: bool, detail_message: str)
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
    local_model: str = LOCAL_MODEL_NAME
) -> tuple:
    """
    Performs Dual-Context Audit synthesis.
    Tries Cloud LLM first; if cloud call fails or rate-limits (429), falls back to local Ollama.
    
    Returns:
        tuple: (response_markdown_text, provider_badge_name)
    """
    # 1. Format Context Inputs
    doc_str = ""
    if doc_context:
        for item in doc_context:
            doc_str += f"- [{item['source']} | Chunk {item['chunk_index']}]: {item['text']}\n"
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
        "- Highlight any conflicts, updates, or omissions between uploaded documents and live web facts."
    )

    user_prompt = (
        f"USER QUERY: {query}\n\n"
        f"TIME-SENSITIVITY DETECTED: {'Yes' if is_time_sensitive else 'No'}\n\n"
        f"--- SOURCE 1: UPLOADED MULTI-MODAL CONTEXT ---\n{doc_str}\n\n"
        f"--- SOURCE 2: LIVE WEB SEARCH VERIFICATION ---\n{web_str}\n\n"
        "Please perform the Dual-Context Audit now."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    cloud_error = ""

    # 2. Primary Execution: Cloud LLM
    if cloud_api_key and cloud_api_key.strip():
        try:
            cloud_client = OpenAI(
                api_key=cloud_api_key.strip(),
                base_url=cloud_base_url.strip(),
                timeout=30.0
            )
            res = cloud_client.chat.completions.create(
                model=cloud_model.strip(),
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
                timeout=30.0
            )
            output = res.choices[0].message.content
            provider_label = "NVIDIA Cloud" if "nvidia" in cloud_base_url else "Cloud LLM (OpenRouter)"
            return output, f"Powered by {provider_label}"
        except Exception as e:
            cloud_error = str(e)
            print(f"[LLM Router Warning] Cloud LLM failed ({cloud_error}). Attempting Local Ollama Fallback...")

    # 3. Secondary Execution: Fallback to Local Ollama
    if check_ollama_online(ollama_base_url):
        try:
            local_client = OpenAI(
                api_key="ollama",
                base_url=ollama_base_url,
                timeout=15.0
            )
            res = local_client.chat.completions.create(
                model=local_model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
                timeout=15.0
            )
            output = res.choices[0].message.content
            return output, "Powered by Local Ollama (Fallback Active)"
        except Exception as e:
            ollama_err = str(e)
            if cloud_error:
                return (
                    f"🚨 **Cloud LLM Error**: `{cloud_error}`\n\n"
                    f"🚨 **Local Ollama Fallback Error**: `{ollama_err}`",
                    "API Execution Failed"
                )
            return f"🚨 **Local Ollama Error**: `{ollama_err}`", "Local Ollama Error"

    # 4. If Cloud failed & Ollama offline:
    if cloud_error:
        return (
            f"🚨 **Cloud LLM API Error**: `{cloud_error}`\n\n"
            "Please check your API key, Base URL, or Model Name in the sidebar.",
            "Cloud API Error"
        )

    return (
        "⚠️ **LLM Service Unavailable**:\n"
        "- Please enter a valid Cloud API Key (OpenRouter / NVIDIA NIM) in the sidebar **OR**\n"
        "- Ensure local Ollama is running (`ollama serve`) at `http://localhost:11434`.",
        "No Active LLM Provider"
    )
