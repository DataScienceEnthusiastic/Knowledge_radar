"""
Document & Multi-Modal Parsers and Chunking Engines.

Provides ingestion interfaces for PDFs, images, YouTube transcripts, and webpages,
alongside sliding-window and sentence-aware chunking strategies.
"""

import io
import os
import re
from typing import Any, Dict, List, Optional, Union

from config import CHUNK_SIZE, CHUNK_OVERLAP
from modules.extractors import (
    ExtractionResult,
    ExtractionOptions,
    YouTubeExtractor,
    extract_source,
    extract_many,
    store_in_chromadb,
    store_in_sqlite,
)


def parse_pdf(pdf_file) -> str:
    """
    Extracts text, headings, and tables from an uploaded PDF file or path.
    Backward-compatible string interface.
    """
    try:
        res = extract_source(pdf_file, source_type="pdf")
        if res.errors:
            return f"Error parsing PDF: {res.errors[0].message}"
        text = res.to_text()
        return text if text.strip() else "PDF contains no extractable text."
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"


def extract_youtube_video_id(url: str) -> str:
    """Extracts 11-character YouTube video ID."""
    return YouTubeExtractor().extract_video_id(url) or ""


def parse_youtube(url: str) -> str:
    """
    Fetches transcript for a YouTube video URL and formats it with [MM:SS] timestamps.
    Backward-compatible string interface.
    """
    try:
        res = extract_source(url, source_type="youtube")
        if res.errors:
            return f"Error fetching YouTube transcript: {res.errors[0].message}"
        if not res.transcript_segments:
            return "No transcript text found for this video (Subtitles may be disabled)."
        return res.to_text()
    except Exception as e:
        return f"Error fetching YouTube transcript: {str(e)}"


def parse_url(url: str) -> str:
    """
    Extracts article content, section hierarchy, and tables from a Web URL safely.
    Backward-compatible string interface.
    """
    try:
        res = extract_source(url, source_type="webpage")
        if res.errors:
            return f"Error fetching web URL: {res.errors[0].message}"
        text = res.to_text()
        return text if text.strip() else "Web URL contains no extractable body text."
    except Exception as e:
        return f"Error fetching web URL: {str(e)}"


def parse_image(image_file) -> str:
    """
    Extracts OCR text from an uploaded image file or path using Pillow and Tesseract.
    Backward-compatible string interface.
    """
    try:
        res = extract_source(image_file, source_type="image")
        if res.errors:
            return f"Error processing image OCR: {res.errors[0].message}"
        text = res.to_text()
        if not text.strip():
            for w in res.extraction_warnings:
                if "TESSERACT_BINARY_MISSING" in w.code:
                    return f"Error: {w.message}"
            return "No readable text detected in the image."
        return text
    except Exception as e:
        return f"Error processing image OCR: {str(e)}"


def extract_structured(
    source: Union[str, bytes, io.BytesIO],
    source_type: Optional[str] = None,
    options: Optional[ExtractionOptions] = None,
    filename: Optional[str] = None,
) -> ExtractionResult:
    """
    Direct structured extraction returning the full ExtractionResult object
    with content_blocks, tables, figures, and transcript_segments.
    """
    return extract_source(source, source_type=source_type, options=options, filename=filename)


def extract_structure_chunks(
    source: Union[str, bytes, io.BytesIO],
    source_type: Optional[str] = None,
    filename: Optional[str] = None,
    max_chunk_chars: int = CHUNK_SIZE,
    chunk_overlap_chars: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Extracts documents into structure-aware RAG chunks with intact table representations,
    visual figure captions, and hierarchical section paths.
    """
    result = extract_source(source, source_type=source_type, filename=filename)
    return result.to_rag_chunks(
        max_chunk_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars
    )


def chunk_text(
    text: str,
    source_name: str = "Document",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Splits text into overlapping sliding-window character chunks for vector embedding.
    Guards against non-positive stepping when chunk_overlap >= chunk_size.
    """
    if not text or not text.strip():
        return []

    text_len = len(text)
    chunks = []
    start = 0
    chunk_idx = 1
    step = max(1, chunk_size - chunk_overlap)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_content = text[start:end].strip()

        if chunk_content:
            chunks.append({
                "text": chunk_content,
                "source": source_name,
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1

        start += step
        if start >= text_len:
            break

    return chunks


def chunk_sentences(
    text: str,
    source_name: str = "Document",
    max_chunk_chars: int = CHUNK_SIZE,
    overlap_sentences: int = 1,
) -> List[Dict[str, Any]]:
    """
    Splits text along sentence and paragraph boundaries rather than blind character counts.
    Preserves complete grammatical units and prevents truncation of numbers or entities.
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs, then sentences
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    all_sentences = []
    for p in paragraphs:
        sents = [s.strip() for s in sentence_pattern.split(p) if s.strip()]
        all_sentences.extend(sents)

    if not all_sentences:
        return chunk_text(text, source_name=source_name, chunk_size=max_chunk_chars)

    chunks = []
    chunk_idx = 1
    current_sentences = []
    current_length = 0

    for sent in all_sentences:
        sent_len = len(sent)
        # If adding sentence exceeds limit and we already have content:
        if current_length + sent_len > max_chunk_chars and current_sentences:
            chunk_text_body = " ".join(current_sentences).strip()
            chunks.append({
                "text": chunk_text_body,
                "source": source_name,
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1

            # Retain overlap sentences for context continuity
            if overlap_sentences > 0:
                current_sentences = current_sentences[-overlap_sentences:]
                current_length = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
            else:
                current_sentences = []
                current_length = 0

        current_sentences.append(sent)
        current_length += sent_len + 1

    if current_sentences:
        chunks.append({
            "text": " ".join(current_sentences).strip(),
            "source": source_name,
            "chunk_index": chunk_idx,
        })

    return chunks
