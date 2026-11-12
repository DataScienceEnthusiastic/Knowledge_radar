"""
Multimodal Document Extraction Engine for Knowledge Radar.

Modular extractors:
- PDFExtractor: PDF extraction with layout, tables, and OCR fallback
- ImageExtractor: Standalone image extraction with OCR and vision adapters
- WebpageExtractor: Web article extraction with SSRF security and DOM hierarchy
- YouTubeExtractor: YouTube video transcript extraction with timestamps
"""
from __future__ import annotations

import io
import os
from typing import Any, List, Optional, Union

from modules.extractors.base import BaseExtractor
from modules.extractors.models import (
    BoundingBox,
    SourceLocation,
    ContentBlock,
    TableElement,
    FigureElement,
    LinkElement,
    TranscriptSegment,
    DocumentMetadata,
    ExtractionWarning,
    ExtractionError,
    ExtractionStatistics,
    ExtractionResult,
    VisionModelAdapter,
    AudioTranscriptionAdapter,
    ExtractionOptions,
)
from modules.extractors.utils import (
    generate_document_id,
    generate_block_id,
    normalize_text,
    validate_and_guard_url,
    detect_source_type,
    _autodetect_tesseract,
)
from modules.extractors.pdf_extractor import PDFExtractor
from modules.extractors.image_extractor import ImageExtractor
from modules.extractors.web_extractor import WebpageExtractor
from modules.extractors.youtube_extractor import YouTubeExtractor
from modules.extractors.storage import store_in_chromadb, store_in_sqlite


def extract_source(
    source: Union[str, bytes, io.BytesIO],
    source_type: Optional[str] = None,
    options: Optional[ExtractionOptions] = None,
    filename: Optional[str] = None,
) -> ExtractionResult:
    """
    Main entry point for document extraction.
    Autodetects type (pdf, image, webpage, youtube) if not specified.
    """
    opts = options or ExtractionOptions()
    resolved_type = source_type or detect_source_type(source, filename=filename)

    if resolved_type == "pdf":
        name = filename or (os.path.basename(source) if isinstance(source, str) else "document.pdf")
        return PDFExtractor(options=opts).extract(source, filename=name)

    elif resolved_type == "image":
        name = filename or (os.path.basename(source) if isinstance(source, str) else "image.png")
        return ImageExtractor(options=opts).extract(source, filename=name)

    elif resolved_type == "webpage":
        if not isinstance(source, str):
            raise ValueError("Webpage source must be a string URL.")
        return WebpageExtractor(options=opts).extract(source)

    elif resolved_type == "youtube":
        if not isinstance(source, str):
            raise ValueError("YouTube source must be a string URL.")
        return YouTubeExtractor(options=opts).extract(source)

    else:
        doc_id = generate_document_id("unknown", str(source)[:100])
        res = ExtractionResult(
            document_id=doc_id,
            source_id=str(source)[:100],
            source_type="unknown",
            original_source=str(source)[:100],
            canonical_source=str(source)[:100],
        )
        res.errors.append(ExtractionError(
            code="UNSUPPORTED_TYPE", message=f"Could not identify source format '{resolved_type}'."
        ))
        res.extraction_status = "failed"
        return res


def extract_many(
    sources: List[Union[str, bytes, io.BytesIO]],
    options: Optional[ExtractionOptions] = None,
) -> List[ExtractionResult]:
    """Batch extraction interface."""
    return [extract_source(src, options=options) for src in sources]


__all__ = [
    "BaseExtractor",
    "BoundingBox",
    "SourceLocation",
    "ContentBlock",
    "TableElement",
    "FigureElement",
    "LinkElement",
    "TranscriptSegment",
    "DocumentMetadata",
    "ExtractionWarning",
    "ExtractionError",
    "ExtractionStatistics",
    "ExtractionResult",
    "VisionModelAdapter",
    "AudioTranscriptionAdapter",
    "ExtractionOptions",
    "generate_document_id",
    "generate_block_id",
    "normalize_text",
    "validate_and_guard_url",
    "detect_source_type",
    "_autodetect_tesseract",
    "PDFExtractor",
    "ImageExtractor",
    "WebpageExtractor",
    "YouTubeExtractor",
    "store_in_chromadb",
    "store_in_sqlite",
    "extract_source",
    "extract_many",
]
