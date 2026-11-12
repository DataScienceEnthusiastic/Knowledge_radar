"""
PDF document extraction engine.
Supports digital PDFs, table detection, heading hierarchy, and OCR fallback.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz as pymupdf
        HAS_PYMUPDF = True
    except ImportError:
        pymupdf = None
        HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    pdfplumber = None
    HAS_PDFPLUMBER = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    pypdf = None
    HAS_PYPDF = False

try:
    from PIL import Image, ImageOps
    HAS_PILLOW = True
except ImportError:
    Image = None
    ImageOps = None
    HAS_PILLOW = False

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    pytesseract = None
    HAS_PYTESSERACT = False

from modules.extractors.base import BaseExtractor
from modules.extractors.models import (
    BoundingBox,
    ContentBlock,
    DocumentMetadata,
    ExtractionError,
    ExtractionOptions,
    ExtractionResult,
    ExtractionStatistics,
    ExtractionWarning,
    FigureElement,
    LinkElement,
    SourceLocation,
    TableElement,
)
from modules.extractors.utils import generate_block_id, generate_document_id, normalize_text

logger = logging.getLogger("document_extractor.pdf")

class PDFExtractor(BaseExtractor):
    """
    Extracts text, hierarchy, tables, and images from PDFs.
    Uses font-size comparison to infer headings and build a hierarchical section_path.
    """

    def __init__(self, options: Optional[ExtractionOptions] = None):
        self.options = options or ExtractionOptions()

    def extract(self, source: Union[str, bytes, io.BytesIO], filename: str = "document.pdf") -> ExtractionResult:
        start_time = time.time()
        file_bytes = self._read_bytes(source)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        canonical = os.path.abspath(source) if isinstance(source, str) and os.path.exists(source) else filename
        doc_id = generate_document_id("pdf", canonical, file_bytes)

        metadata = DocumentMetadata(
            original_filename=filename,
            file_extension=".pdf",
            mime_type="application/pdf",
            file_size_bytes=len(file_bytes),
            file_hash=file_hash,
            canonical_url=canonical,
        )

        result = ExtractionResult(
            document_id=doc_id,
            source_id=canonical,
            source_type="pdf",
            original_source=str(source) if isinstance(source, str) else filename,
            canonical_source=canonical,
            document_metadata=metadata,
        )

        if HAS_PYMUPDF:
            self._parse_pymupdf(file_bytes, result)
        elif HAS_PDFPLUMBER:
            self._parse_pdfplumber(file_bytes, result)
        elif HAS_PYPDF:
            self._parse_pypdf(file_bytes, result)
        else:
            result.errors.append(ExtractionError(
                code="NO_PDF_PARSER",
                message="No PDF parser library installed (pymupdf, pdfplumber, pypdf).",
                suggested_action="pip install pymupdf pdfplumber",
            ))
            result.extraction_status = "failed"
            return result

        result.extraction_statistics = ExtractionStatistics(
            page_count=result.document_metadata.page_count or 0,
            content_block_count=len(result.content_blocks),
            table_count=len(result.tables),
            figure_count=len(result.figures),
            link_count=len(result.links),
            ocr_pages_count=sum(1 for b in result.content_blocks if b.extraction_method == "ocr"),
            extraction_duration_seconds=round(time.time() - start_time, 3),
            warning_count=len(result.extraction_warnings),
            error_count=len(result.errors),
        )

        result.extraction_status = "failed" if result.errors else (
            "success_with_warnings" if result.extraction_warnings else "success"
        )
        return result

    def _read_bytes(self, source: Union[str, bytes, io.BytesIO]) -> bytes:
        if isinstance(source, bytes):
            return source
        if isinstance(source, str):
            with open(source, "rb") as f:
                return f.read()
        if hasattr(source, "read"):
            pos = source.tell() if hasattr(source, "tell") else 0
            b = source.read()
            if hasattr(source, "seek"):
                source.seek(pos)
            return b
        raise ValueError(f"Cannot read source of type {type(source)}")

    def _parse_pymupdf(self, file_bytes: bytes, result: ExtractionResult):
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            result.errors.append(ExtractionError(code="PDF_OPEN_ERROR", message=f"Failed to open PDF: {e}"))
            return

        result.document_metadata.page_count = len(doc)
        pdf_meta = doc.metadata or {}
        result.title = pdf_meta.get("title") or None
        result.document_metadata.title = result.title
        result.document_metadata.author = pdf_meta.get("author") or None

        max_p = self.options.max_pages or len(doc)
        section_stack: List[str] = []
        reading_order = 1

        for page_idx in range(min(len(doc), max_p)):
            page_num = page_idx + 1
            page = doc[page_idx]
            w, h = page.rect.width, page.rect.height

            # 1. Scanned page check
            raw_text = page.get_text()
            if len(raw_text.strip()) < self.options.ocr_scanned_threshold:
                if self._try_ocr_page(page, page_num, result, section_stack, reading_order):
                    reading_order += 1
                    continue
                else:
                    result.extraction_warnings.append(ExtractionWarning(
                        code="SCANNED_PAGE_NO_OCR",
                        message=f"Page {page_num} appears scanned and OCR was unavailable.",
                        page_number=page_num,
                    ))

            # 2. Extract structured text blocks
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])

            # Compute median font size on the page so we can spot headings naturally
            font_sizes = []
            for b in blocks:
                if b.get("type") == 0:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("text", "").strip():
                                font_sizes.append(span.get("size", 10.0))
            median_font = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0

            for b in blocks:
                # Text Block
                if b.get("type") == 0:
                    bbox = b.get("bbox", [0, 0, 0, 0])
                    b_box = BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], page_width=w, page_height=h)

                    lines = []
                    max_size = 0.0
                    is_bold = False

                    for line in b.get("lines", []):
                        line_str = "".join(span.get("text", "") for span in line.get("spans", []))
                        for span in line.get("spans", []):
                            sz = span.get("size", 0.0)
                            if sz > max_size:
                                max_size = sz
                            if (span.get("flags", 0) & 2) or "bold" in span.get("font", "").lower():
                                is_bold = True
                        if line_str.strip():
                            lines.append(line_str)

                    full_text = "\n".join(lines).strip()
                    if not full_text:
                        continue

                    # Heading heuristics:
                    # Larger than median font OR bold OR numbered prefix like "1.2 Intro",
                    # reasonably short, and doesn't end like a sentence with a period.
                    has_numbered_prefix = bool(re.match(r"^\d+(\.\d+)*\s+[A-Z]", full_text))
                    is_heading = (
                        (max_size >= median_font * 1.10) or is_bold or has_numbered_prefix
                    ) and len(full_text) < 160 and not full_text.endswith(".")

                    elem_type = "paragraph"
                    if is_heading:
                        elem_type = "heading"
                        clean_h = re.sub(r"\s+", " ", full_text).strip()
                        # Update section hierarchy stack
                        if max_size >= median_font * 1.35:
                            section_stack = [clean_h]
                        else:
                            if len(section_stack) > 1:
                                section_stack = [section_stack[0], clean_h]
                            else:
                                section_stack.append(clean_h)
                    elif re.match(r"^(\d+[\.\)]|[\u2022\-\*])\s+", full_text):
                        elem_type = "list"

                    bid = generate_block_id(result.document_id, elem_type, page_num, reading_order, full_text)
                    result.content_blocks.append(ContentBlock(
                        block_id=bid,
                        document_id=result.document_id,
                        element_type=elem_type,
                        text=full_text,
                        normalized_text=normalize_text(full_text),
                        page_number=page_num,
                        section_path=list(section_stack),
                        reading_order=reading_order,
                        bounding_box=b_box,
                        source_location=SourceLocation(page_number=page_num, bounding_box=b_box),
                        extraction_method="pymupdf_native",
                        metadata={"font_size": round(max_size, 2), "is_bold": is_bold},
                    ))
                    reading_order += 1

                # Embedded Figure Block
                elif b.get("type") == 1 and self.options.extract_figures:
                    bbox = b.get("bbox", [0, 0, 0, 0])
                    b_box = BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], page_width=w, page_height=h)
                    fig_id = f"fig_{result.document_id[:8]}_{page_num}_{len(result.figures) + 1}"

                    caption, is_inferred = self._find_nearby_caption(page, bbox, w, h, is_table=False)
                    result.figures.append(FigureElement(
                        figure_id=fig_id,
                        document_id=result.document_id,
                        page_number=page_num,
                        bounding_box=b_box,
                        caption=caption,
                        caption_is_inferred=is_inferred,
                        section_heading=section_stack[-1] if section_stack else None,
                        image_format=b.get("ext", "png").upper(),
                        image_dimensions=(int(b_box.width), int(b_box.height)),
                        extraction_method="pymupdf_image_block",
                    ))

            # 3. Tables via TableFinder
            if self.options.extract_tables and hasattr(page, "find_tables"):
                self._extract_tables(page, page_num, result, section_stack)

            # 4. Hyperlinks
            if self.options.extract_links:
                for lnk in page.get_links():
                    uri = lnk.get("uri")
                    if uri:
                        lid = f"lnk_{result.document_id[:8]}_{page_num}_{len(result.links) + 1}"
                        result.links.append(LinkElement(
                            link_id=lid,
                            document_id=result.document_id,
                            url=uri,
                            text=uri,
                            page_number=page_num,
                            section_heading=section_stack[-1] if section_stack else None,
                            is_external=uri.startswith(("http://", "https://")),
                        ))

        doc.close()

    def _extract_tables(self, page: Any, page_num: int, result: ExtractionResult, section_stack: List[str]):
        """Extracts structured tables with headers, cells, and RAG representations."""
        try:
            for tab in page.find_tables().tables:
                raw = tab.extract()
                if not raw or len(raw) < 1:
                    continue

                headers = [str(c or "").strip() for c in raw[0]]
                rows = raw[1:] if len(raw) > 1 else []
                uncertain = all(not h or h.isdigit() for h in headers)

                # Markdown table
                md = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
                md += "\n".join("| " + " | ".join(str(c or "").strip() for c in r) + " |" for r in rows)

                # Retrieval text: Row 1: Col1: Val | Col2: Val
                lines = [f"Table (Page {page_num})", f"Columns: {', '.join(headers)}"]
                for i, r in enumerate(rows, 1):
                    pairs = [f"{headers[j]}: {str(c or '').strip()}" for j, c in enumerate(r) if j < len(headers)]
                    lines.append(f"Row {i}: {' | '.join(pairs)}")
                text_repr = "\n".join(lines)

                bbox = tab.bbox
                b_box = BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], page_width=page.rect.width, page_height=page.rect.height)
                tbl_id = f"tbl_{result.document_id[:8]}_{page_num}_{len(result.tables) + 1}"
                caption, is_inferred = self._find_nearby_caption(page, bbox, page.rect.width, page.rect.height, is_table=True)

                table_el = TableElement(
                    table_id=tbl_id,
                    document_id=result.document_id,
                    page_number=page_num,
                    page_range=[page_num],
                    bounding_box=b_box,
                    caption=caption,
                    caption_is_inferred=is_inferred,
                    section_heading=section_stack[-1] if section_stack else None,
                    column_headers=headers,
                    header_detection_uncertain=uncertain,
                    rows=rows,
                    row_count=len(rows),
                    col_count=len(headers),
                    extraction_method="pymupdf_table_finder",
                    markdown_representation=md,
                    text_representation=text_repr,
                )
                result.tables.append(table_el)

                # Also record as ContentBlock for reading order continuity
                result.content_blocks.append(ContentBlock(
                    block_id=tbl_id,
                    document_id=result.document_id,
                    element_type="table",
                    text=md,
                    normalized_text=normalize_text(text_repr),
                    page_number=page_num,
                    section_path=list(section_stack),
                    reading_order=len(result.content_blocks) + 1,
                    bounding_box=b_box,
                    source_location=SourceLocation(page_number=page_num, bounding_box=b_box),
                    extraction_method="pymupdf_table_finder",
                ))
        except Exception as e:
            result.extraction_warnings.append(ExtractionWarning(
                code="TABLE_ERROR", message=f"Table extraction failed on page {page_num}: {e}", page_number=page_num
            ))

    def _find_nearby_caption(
        self, page: Any, bbox: Tuple[float, float, float, float], page_w: float, page_h: float, is_table: bool = False
    ) -> Tuple[Optional[str], bool]:
        """Looks immediately above and below a bounding box for explicit captions."""
        try:
            pattern = (
                r"(?i)^(?:Table|Tab\.)\s*(\d+[A-Za-z0-9_.:-]*)\s*[:.\-—]\s*(.+)"
                if is_table else
                r"(?i)^(?:Figure|Fig\.|Chart|Diagram)\s*(\d+[A-Za-z0-9_.:-]*)\s*[:.\-—]\s*(.+)"
            )
            # 45 points above and below
            above = pymupdf.Rect(bbox[0] - 10, max(0, bbox[1] - 45), bbox[2] + 10, bbox[1] + 5)
            below = pymupdf.Rect(bbox[0] - 10, bbox[3] - 5, bbox[2] + 10, min(page_h, bbox[3] + 45))

            for r in (above, below):
                txt = page.get_text("text", clip=r).strip()
                if txt and re.search(pattern, txt):
                    return txt.replace("\n", " ").strip(), False
            return None, False
        except Exception:
            return None, False

    def _try_ocr_page(self, page: Any, page_num: int, result: ExtractionResult, section_stack: List[str], order: int) -> bool:
        if not (self.options.enable_ocr and HAS_PYTESSERACT and HAS_PILLOW):
            return False
        try:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=self.options.ocr_languages).strip()
            if not text:
                return False

            result.document_metadata.ocr_applied = True
            result.document_metadata.ocr_engine = f"Tesseract ({self.options.ocr_languages})"
            bid = generate_block_id(result.document_id, "paragraph", page_num, order, text)

            result.content_blocks.append(ContentBlock(
                block_id=bid,
                document_id=result.document_id,
                element_type="paragraph",
                text=text,
                normalized_text=normalize_text(text),
                page_number=page_num,
                section_path=list(section_stack),
                reading_order=order,
                source_location=SourceLocation(page_number=page_num),
                extraction_method="ocr",
            ))
            return True
        except Exception as e:
            result.extraction_warnings.append(ExtractionWarning(
                code="OCR_FAILED", message=f"OCR failed on page {page_num}: {e}", page_number=page_num
            ))
            return False

    def _parse_pdfplumber(self, file_bytes: bytes, result: ExtractionResult):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            result.document_metadata.page_count = len(pdf.pages)
            order = 1
            for idx, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    bid = generate_block_id(result.document_id, "paragraph", idx + 1, order, text)
                    result.content_blocks.append(ContentBlock(
                        block_id=bid,
                        document_id=result.document_id,
                        element_type="paragraph",
                        text=text,
                        normalized_text=normalize_text(text),
                        page_number=idx + 1,
                        reading_order=order,
                        extraction_method="pdfplumber_native",
                    ))
                    order += 1

    def _parse_pypdf(self, file_bytes: bytes, result: ExtractionResult):
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        result.document_metadata.page_count = len(reader.pages)
        order = 1
        for idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                bid = generate_block_id(result.document_id, "paragraph", idx + 1, order, text)
                result.content_blocks.append(ContentBlock(
                    block_id=bid,
                    document_id=result.document_id,
                    element_type="paragraph",
                    text=text,
                    normalized_text=normalize_text(text),
                    page_number=idx + 1,
                    reading_order=order,
                    extraction_method="pypdf_fallback",
                ))
                order += 1
