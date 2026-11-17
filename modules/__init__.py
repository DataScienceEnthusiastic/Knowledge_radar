"""
Knowledge Radar Core Modules.

Organized into connected architectural domains:
- extractors: Multi-modal document ingestion & parsing
- retrieval: Feature hashing, vector storage & similarity retrieval
- reasoning: LLM model routing & live web fact-checking
- auditing: Discrete claim extraction, grounding verification & drift tracking
"""

from modules.extractors import (
    BaseExtractor,
    PDFExtractor,
    ImageExtractor,
    WebpageExtractor,
    YouTubeExtractor,
    extract_source,
    extract_many,
    parse_pdf,
    parse_image,
    parse_youtube,
    parse_url,
    chunk_text,
    chunk_sentences,
)
from modules.retrieval import KnowledgeVectorStore
from modules.reasoning import (
    check_ollama_online,
    is_time_sensitive_local,
    generate_response,
    test_cloud_connection,
    fetch_live_web_context,
)
from modules.auditing import (
    evaluate_turn,
    audit_claims_and_evidence,
    SessionManager,
    SessionState,
)

__all__ = [
    "BaseExtractor",
    "PDFExtractor",
    "ImageExtractor",
    "WebpageExtractor",
    "YouTubeExtractor",
    "extract_source",
    "extract_many",
    "parse_pdf",
    "parse_image",
    "parse_youtube",
    "parse_url",
    "chunk_text",
    "chunk_sentences",
    "KnowledgeVectorStore",
    "check_ollama_online",
    "is_time_sensitive_local",
    "generate_response",
    "test_cloud_connection",
    "fetch_live_web_context",
    "evaluate_turn",
    "audit_claims_and_evidence",
    "SessionManager",
    "SessionState",
]
