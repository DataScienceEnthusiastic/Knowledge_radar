"""
Document & Multi-Modal Parsers and Chunking Engines (Forwarding Facade).

Re-exports parsing and chunking interfaces from the unified modules.extractors package.
"""
from modules.extractors.parser import (
    parse_pdf,
    extract_youtube_video_id,
    parse_youtube,
    parse_url,
    parse_image,
    extract_structured,
    extract_structure_chunks,
    chunk_text,
    chunk_sentences,
)

__all__ = [
    "parse_pdf",
    "extract_youtube_video_id",
    "parse_youtube",
    "parse_url",
    "parse_image",
    "extract_structured",
    "extract_structure_chunks",
    "chunk_text",
    "chunk_sentences",
]
