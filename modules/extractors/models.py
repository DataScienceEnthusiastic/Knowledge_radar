"""
Data models, entities, and protocols for document extraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

@dataclass
class BoundingBox:
    """
    Coordinates in document points (PDF: 72 dpi, origin at top-left).
    Matches PyMuPDF's coordinate system [x0, y0, x1, y1].
    """
    x0: float
    y0: float
    x1: float
    y1: float
    coord_origin: str = "top-left"
    units: str = "pt"
    page_width: Optional[float] = None
    page_height: Optional[float] = None

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def as_list(self) -> List[float]:
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["width"] = round(self.width, 2)
        d["height"] = round(self.height, 2)
        return d


@dataclass
class SourceLocation:
    """Provenance pointer back to where an item came from."""
    page_number: Optional[int] = None  # 1-based index
    bounding_box: Optional[BoundingBox] = None
    character_start: Optional[int] = None
    character_end: Optional[int] = None
    line_number: Optional[int] = None
    source_uri: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.bounding_box:
            d["bounding_box"] = self.bounding_box.to_dict()
        return d


@dataclass
class ContentBlock:
    """
    An atomic text element (paragraph, heading, list item, or code snippet).
    Maintains hierarchical section path (e.g. ['1. Intro', '1.1 Setup']) for RAG context.
    """
    block_id: str
    document_id: str
    element_type: str  # paragraph | heading | list | table | figure | caption | transcript | code | other
    text: str
    normalized_text: str
    page_number: Optional[int] = None
    section_path: List[str] = field(default_factory=list)
    reading_order: int = 0
    parent_block_id: Optional[str] = None
    source_location: Optional[SourceLocation] = None
    bounding_box: Optional[BoundingBox] = None
    source_element_id: Optional[str] = None
    related_block_ids: List[str] = field(default_factory=list)
    extraction_method: str = "native"
    confidence: Optional[float] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.bounding_box:
            d["bounding_box"] = self.bounding_box.to_dict()
        if self.source_location:
            d["source_location"] = self.source_location.to_dict()
        return d


@dataclass
class TableElement:
    """
    A detected table.
    We keep both structured rows and two string formats:
    - markdown_representation: good for LLM reading
    - text_representation: 'Col: Val | Col: Val' row-by-row format, which performs
      significantly better for dense vector similarity retrieval.
    """
    table_id: str
    document_id: str
    page_number: Optional[int] = None
    page_range: List[int] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    caption: Optional[str] = None
    caption_is_inferred: bool = False
    section_heading: Optional[str] = None
    column_headers: List[str] = field(default_factory=list)
    header_detection_uncertain: bool = False
    rows: List[List[Any]] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    merged_cells: List[Dict[str, Any]] = field(default_factory=list)
    extraction_method: str = "auto"
    confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    image_reference: Optional[str] = None
    markdown_representation: str = ""
    text_representation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.bounding_box:
            d["bounding_box"] = self.bounding_box.to_dict()
        return d


@dataclass
class FigureElement:
    """An image, diagram, or chart with optional OCR and vision analysis."""
    figure_id: str
    document_id: str
    page_number: Optional[int] = None
    bounding_box: Optional[BoundingBox] = None
    caption: Optional[str] = None
    caption_is_inferred: bool = False
    section_heading: Optional[str] = None
    image_reference: Optional[str] = None
    image_format: Optional[str] = None
    image_dimensions: Optional[Tuple[int, int]] = None
    related_block_ids: List[str] = field(default_factory=list)
    extraction_method: str = "embedded_image"
    visual_analysis_performed: bool = False
    visual_analysis_status: str = "not_analyzed"
    visual_description: Optional[str] = None
    structured_chart_data: Optional[Dict[str, Any]] = None
    ocr_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.bounding_box:
            d["bounding_box"] = self.bounding_box.to_dict()
        return d


@dataclass
class LinkElement:
    link_id: str
    document_id: str
    url: str
    text: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    is_external: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptSegment:
    """Timestamped segment from a YouTube video or audio file."""
    segment_id: str
    document_id: str
    text: str
    start_time_seconds: float = 0.0
    end_time_seconds: float = 0.0
    duration_seconds: float = 0.0
    language: str = "en"
    transcript_source: str = "captions"  # captions | automatic_captions | speech_recognition
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentMetadata:
    original_filename: Optional[str] = None
    file_extension: Optional[str] = None
    mime_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_hash: Optional[str] = None  # SHA-256 for reproducibility
    page_count: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    source_url: Optional[str] = None
    canonical_url: Optional[str] = None
    publisher: Optional[str] = None
    website_name: Optional[str] = None
    channel_name: Optional[str] = None
    duration_seconds: Optional[float] = None
    extraction_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    parser_name: str = "KnowledgeRadar-Extractor"
    parser_version: str = "2.5.0"
    ocr_applied: bool = False
    ocr_engine: Optional[str] = None
    detected_language: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionWarning:
    code: str
    message: str
    source_identifier: Optional[str] = None
    page_number: Optional[int] = None
    block_id: Optional[str] = None
    stage: str = "extraction"
    recoverable: bool = True
    suggested_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionError:
    code: str
    message: str
    source_identifier: Optional[str] = None
    page_number: Optional[int] = None
    stage: str = "extraction"
    recoverable: bool = False
    suggested_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionStatistics:
    page_count: int = 0
    content_block_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    link_count: int = 0
    transcript_segment_count: int = 0
    ocr_pages_count: int = 0
    extraction_duration_seconds: float = 0.0
    warning_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    """
    Top-level output containing everything extracted from a source.
    Offers helper methods to export JSON, linear text, or downstream RAG chunks.
    """
    document_id: str
    source_id: str
    source_type: str  # pdf | image | webpage | youtube | unknown
    original_source: str
    canonical_source: str
    title: Optional[str] = None
    document_metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    content_blocks: List[ContentBlock] = field(default_factory=list)
    tables: List[TableElement] = field(default_factory=list)
    figures: List[FigureElement] = field(default_factory=list)
    links: List[LinkElement] = field(default_factory=list)
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    extraction_status: str = "success"  # success | success_with_warnings | partial_success | failed
    extraction_warnings: List[ExtractionWarning] = field(default_factory=list)
    errors: List[ExtractionError] = field(default_factory=list)
    extraction_statistics: ExtractionStatistics = field(default_factory=ExtractionStatistics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "original_source": self.original_source,
            "canonical_source": self.canonical_source,
            "title": self.title,
            "document_metadata": self.document_metadata.to_dict(),
            "content_blocks": [b.to_dict() for b in self.content_blocks],
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "links": [l.to_dict() for l in self.links],
            "transcript_segments": [s.to_dict() for s in self.transcript_segments],
            "extraction_status": self.extraction_status,
            "extraction_warnings": [w.to_dict() for w in self.extraction_warnings],
            "errors": [e.to_dict() for e in self.errors],
            "extraction_statistics": self.extraction_statistics.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """
        Builds a human-readable linear transcript of the document in natural reading order.
        """
        lines = []
        if self.title:
            lines.append(f"# {self.title}\n")

        # YouTube transcript formatting
        if self.transcript_segments:
            for s in self.transcript_segments:
                m, sec = int(s.start_time_seconds) // 60, int(s.start_time_seconds) % 60
                lines.append(f"[{m:02d}:{sec:02d}] {s.text}")
            return "\n".join(lines)

        # Standard document formatting
        last_section = None
        for b in self.content_blocks:
            sec_header = " > ".join(b.section_path) if b.section_path else None
            if sec_header and sec_header != last_section:
                lines.append(f"\n## {sec_header}\n")
                last_section = sec_header

            if b.element_type == "heading":
                lines.append(f"\n### {b.text}\n")
            elif b.element_type == "list":
                lines.append(f"- {b.text}")
            elif b.element_type == "table":
                lines.append(f"\n{b.text}\n")
            elif b.element_type == "figure":
                lines.append(f"\n[Figure: {b.text}]\n")
            else:
                lines.append(b.text)

        return "\n\n".join(lines).strip()

    def to_rag_chunks(
        self,
        max_chunk_chars: int = 800,
        chunk_overlap_chars: int = 100,
        include_tables_as_chunks: bool = True,
        include_figures_as_chunks: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Converts the extracted elements into retrieval chunks with rich metadata
        ready for ChromaDB, SQLite, or other vector stores.
        """
        chunks = []
        chunk_idx = 1

        # 1. YouTube Transcripts -> Time-windowed chunks
        if self.transcript_segments:
            buf = []
            start_s = self.transcript_segments[0].start_time_seconds
            end_s = start_s

            for seg in self.transcript_segments:
                buf.append(seg.text)
                end_s = seg.end_time_seconds
                joined = " ".join(buf)

                if len(joined) >= max_chunk_chars:
                    s_str = f"{int(start_s)//60:02d}:{int(start_s)%60:02d}"
                    e_str = f"{int(end_s)//60:02d}:{int(end_s)%60:02d}"
                    tag = f"[{s_str} - {e_str}]"
                    chunks.append({
                        "id": f"{self.document_id}_yt_{chunk_idx:04d}",
                        "document_id": self.document_id,
                        "chunk_index": chunk_idx,
                        "element_type": "transcript",
                        "text": f"{tag} {joined}",
                        "page_number": None,
                        "section_path": "Video Transcript",
                        "metadata": {
                            "document_id": self.document_id,
                            "source_type": self.source_type,
                            "canonical_source": self.canonical_source,
                            "title": self.title or "YouTube Video",
                            "start_time_seconds": start_s,
                            "end_time_seconds": end_s,
                            "time_range": tag,
                        }
                    })
                    chunk_idx += 1
                    buf = []
                    start_s = end_s

            if buf:
                joined = " ".join(buf)
                s_str = f"{int(start_s)//60:02d}:{int(start_s)%60:02d}"
                e_str = f"{int(end_s)//60:02d}:{int(end_s)%60:02d}"
                tag = f"[{s_str} - {e_str}]"
                chunks.append({
                    "id": f"{self.document_id}_yt_{chunk_idx:04d}",
                    "document_id": self.document_id,
                    "chunk_index": chunk_idx,
                    "element_type": "transcript",
                    "text": f"{tag} {joined}",
                    "page_number": None,
                    "section_path": "Video Transcript",
                    "metadata": {
                        "document_id": self.document_id,
                        "source_type": self.source_type,
                        "canonical_source": self.canonical_source,
                        "title": self.title or "YouTube Video",
                        "start_time_seconds": start_s,
                        "end_time_seconds": end_s,
                        "time_range": tag,
                    }
                })
            return chunks

        # 2. Structured Tables as dedicated high-precision chunks
        if include_tables_as_chunks:
            for tbl in self.tables:
                txt = tbl.text_representation or tbl.markdown_representation
                if txt.strip():
                    chunks.append({
                        "id": f"{self.document_id}_{tbl.table_id}",
                        "document_id": self.document_id,
                        "chunk_index": chunk_idx,
                        "element_type": "table",
                        "text": txt,
                        "page_number": tbl.page_number,
                        "section_path": tbl.section_heading or "Table",
                        "metadata": {
                            "document_id": self.document_id,
                            "source_type": self.source_type,
                            "canonical_source": self.canonical_source,
                            "title": self.title or "Document",
                            "page_number": tbl.page_number or 1,
                            "caption": tbl.caption or "",
                            "col_count": tbl.col_count,
                            "row_count": tbl.row_count,
                            "is_table": True,
                        }
                    })
                    chunk_idx += 1

        # 3. Figures with OCR or visual description
        if include_figures_as_chunks:
            for fig in self.figures:
                notes = []
                if fig.caption:
                    notes.append(f"Figure Caption: {fig.caption}")
                if fig.visual_description:
                    notes.append(f"Visual Description: {fig.visual_description}")
                if fig.ocr_text:
                    notes.append(f"Detected Text: {fig.ocr_text}")
                if notes:
                    chunks.append({
                        "id": f"{self.document_id}_{fig.figure_id}",
                        "document_id": self.document_id,
                        "chunk_index": chunk_idx,
                        "element_type": "figure",
                        "text": "\n".join(notes),
                        "page_number": fig.page_number,
                        "section_path": fig.section_heading or "Figure",
                        "metadata": {
                            "document_id": self.document_id,
                            "source_type": self.source_type,
                            "canonical_source": self.canonical_source,
                            "title": self.title or "Document",
                            "page_number": fig.page_number or 1,
                            "caption": fig.caption or "",
                            "is_figure": True,
                        }
                    })
                    chunk_idx += 1

        # 4. Standard Content Blocks (paragraphs, headings, lists)
        buf = []
        buf_len = 0
        cur_page = None
        cur_sec = None

        for b in self.content_blocks:
            if b.element_type in ("table", "figure"):
                continue

            sec_path = " > ".join(b.section_path) if b.section_path else ""
            prefix = f"[{sec_path}]\n" if (sec_path and sec_path != cur_sec) else ""
            line = f"{prefix}{b.text}".strip()
            if not line:
                continue

            if buf_len + len(line) > max_chunk_chars and buf:
                full = "\n\n".join(buf)
                chunks.append({
                    "id": f"{self.document_id}_blk_{chunk_idx:04d}",
                    "document_id": self.document_id,
                    "chunk_index": chunk_idx,
                    "element_type": "text",
                    "text": full,
                    "page_number": cur_page,
                    "section_path": cur_sec or "",
                    "metadata": {
                        "document_id": self.document_id,
                        "source_type": self.source_type,
                        "canonical_source": self.canonical_source,
                        "title": self.title or "Document",
                        "page_number": cur_page or 1,
                        "section_path": cur_sec or "",
                    }
                })
                chunk_idx += 1

                # Overlap logic
                if chunk_overlap_chars > 0 and len(full) > chunk_overlap_chars:
                    buf = [full[-chunk_overlap_chars:]]
                    buf_len = len(buf[0])
                else:
                    buf = []
                    buf_len = 0

            buf.append(line)
            buf_len += len(line)
            cur_page = b.page_number
            cur_sec = sec_path

        if buf:
            chunks.append({
                "id": f"{self.document_id}_blk_{chunk_idx:04d}",
                "document_id": self.document_id,
                "chunk_index": chunk_idx,
                "element_type": "text",
                "text": "\n\n".join(buf),
                "page_number": cur_page,
                "section_path": cur_sec or "",
                "metadata": {
                    "document_id": self.document_id,
                    "source_type": self.source_type,
                    "canonical_source": self.canonical_source,
                    "title": self.title or "Document",
                    "page_number": cur_page or 1,
                    "section_path": cur_sec or "",
                }
            })

        return chunks


# ---------------------------------------------------------------------------
# Options & Adapter Protocols
# ---------------------------------------------------------------------------

class VisionModelAdapter(Protocol):
    """Protocol for optional multimodal vision models (GPT-4o, Claude 3.5, Ollama LLaVA)."""
    def describe_visual(
        self, image_bytes: bytes, mime_type: str = "image/png", prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        ...


class AudioTranscriptionAdapter(Protocol):
    """Protocol for optional speech-to-text fallback (Whisper, Groq, etc.)."""
    def transcribe_audio(
        self, audio_filepath: str, language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        ...


@dataclass
class ExtractionOptions:
    """Configurable knobs for extraction behavior."""
    extract_tables: bool = True
    extract_figures: bool = True
    save_figure_images: bool = False
    figures_output_dir: Optional[str] = None
    enable_ocr: bool = True
    ocr_languages: str = "eng"
    ocr_scanned_threshold: float = 40.0  # Chars per page below which OCR is triggered
    enable_vision_analysis: bool = False
    vision_adapter: Optional[VisionModelAdapter] = None
    enable_audio_transcription_fallback: bool = False
    audio_transcription_adapter: Optional[AudioTranscriptionAdapter] = None
    max_pages: Optional[int] = None
    max_file_size_bytes: int = 100 * 1024 * 1024  # 100 MB safety cap
    request_timeout_seconds: float = 15.0
    user_agent: str = "KnowledgeRadar-Bot/2.5 (+https://knowledgeradar.local)"
    block_private_networks: bool = True  # SSRF protection
    extract_links: bool = True

