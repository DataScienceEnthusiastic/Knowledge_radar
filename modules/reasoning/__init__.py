"""
Reasoning, LLM Routing, and Live Web Verification Engine.
"""
from modules.reasoning.llm_router import (
    check_ollama_online,
    is_time_sensitive_local,
    generate_response,
    test_cloud_connection,
)
from modules.reasoning.web_search import fetch_live_web_context

__all__ = [
    "check_ollama_online",
    "is_time_sensitive_local",
    "generate_response",
    "test_cloud_connection",
    "fetch_live_web_context",
]
