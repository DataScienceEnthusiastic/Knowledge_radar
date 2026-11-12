"""
Standalone image extraction engine.
Handles orientation, dimensions, OCR, and vision model adapters.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
    SourceLocation,
)
from modules.extractors.utils import generate_block_id, generate_document_id, normalize_text

logger = logging.getLogger("document_extractor.image")

class ImageExtractor(BaseExtractor):
    """
    Extracts standalone images (PNG, JPEG, TIFF, WebP).
    Corrects EXIF orientation, extracts dimensions, and runs OCR with line bboxes.
    """

    def __init__(self, options: Optional[ExtractionOptions] = None):
        self.options = options or ExtractionOptions()

    def extract(self, source: Union[str, bytes, io.BytesIO], filename: str = "image.png") -> ExtractionResult:
        start_time = time.time()
        file_bytes = self._read_bytes(source)
        canonical = os.path.abspath(source) if isinstance(source, str) and os.path.exists(source) else filename
        doc_id = generate_document_id("image", canonical, file_bytes)

        metadata = DocumentMetadata(
            original_filename=filename,
            file_size_bytes=len(file_bytes),
            file_hash=hashlib.sha256(file_bytes).hexdigest(),
            canonical_url=canonical,
        )

        result = ExtractionResult(
            document_id=doc_id,
            source_id=canonical,
            source_type="image",
            original_source=str(source) if isinstance(source, str) else filename,
            canonical_source=canonical,
            document_metadata=metadata,
        )

        if not HAS_PILLOW:
            result.errors.append(ExtractionError(code="NO_PILLOW", message="Pillow (PIL) is not installed."))
            result.extraction_status = "failed"
            return result

        try:
            img = Image.open(io.BytesIO(file_bytes))
            # Save original format before EXIF transpose replaces the object
            orig_format = img.format
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            img_format = orig_format
            if not img_format and filename:
                ext = os.path.splitext(filename)[1].lstrip(".").upper()
                if ext in ("PNG", "JPEG", "JPG", "WEBP", "TIFF", "BMP"):
                    img_format = "JPEG" if ext == "JPG" else ext
            img_format = img_format or "UNKNOWN"

            w, h = img.size
            result.document_metadata.file_extension = f".{img_format.lower()}"
            result.document_metadata.mime_type = f"image/{img_format.lower()}"

            fig_id = f"fig_{doc_id[:8]}_1_1"
            figure = FigureElement(
                figure_id=fig_id,
                document_id=doc_id,
                page_number=1,
                bounding_box=BoundingBox(x0=0, y0=0, x1=w, y1=h, units="px"),
                image_format=img_format,
                image_dimensions=(w, h),
                extraction_method="pillow_standalone",
            )

            # 1. OCR text recognition
            if self.options.enable_ocr and HAS_PYTESSERACT:
                self._run_ocr(img, figure, result)
            elif not HAS_PYTESSERACT:
                result.extraction_warnings.append(ExtractionWarning(
                    code="TESSERACT_NOT_INSTALLED", message="pytesseract is not installed; OCR skipped."
                ))

            # 2. Vision model adapter (optional)
            if self.options.enable_vision_analysis and self.options.vision_adapter:
                try:
                    vision = self.options.vision_adapter.describe_visual(file_bytes)
                    figure.visual_analysis_performed = True
                    figure.visual_analysis_status = "completed"
                    figure.visual_description = vision.get("description")
                    figure.structured_chart_data = vision
                except Exception as e:
                    figure.visual_analysis_status = "failed"
                    figure.warnings.append(f"Vision adapter error: {e}")
            else:
                figure.visual_analysis_status = "not_configured"

            result.figures.append(figure)

        except Exception as e:
            result.errors.append(ExtractionError(code="IMAGE_DECODE_ERROR", message=f"Failed to decode image: {e}"))

        result.extraction_statistics = ExtractionStatistics(
            page_count=1,
            content_block_count=len(result.content_blocks),
            figure_count=len(result.figures),
            ocr_pages_count=1 if result.document_metadata.ocr_applied else 0,
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

    def _run_ocr(self, img: Any, figure: FigureElement, result: ExtractionResult):
        try:
            data = pytesseract.image_to_data(img, lang=self.options.ocr_languages, output_type=pytesseract.Output.DICT)
            lines: Dict[int, List[str]] = {}
            boxes: Dict[int, List[int]] = {}

            for i, txt in enumerate(data["text"]):
                t = txt.strip()
                if not t:
                    continue
                ln = data["line_num"][i]
                if ln not in lines:
                    lines[ln] = [t]
                    boxes[ln] = [data["left"][i], data["top"][i], data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]]
                else:
                    lines[ln].append(t)
                    boxes[ln][2] = max(boxes[ln][2], data["left"][i] + data["width"][i])
                    boxes[ln][3] = max(boxes[ln][3], data["top"][i] + data["height"][i])

            all_lines = []
            order = 1
            for ln, words in sorted(lines.items()):
                line_str = " ".join(words).strip()
                if not line_str:
                    continue
                all_lines.append(line_str)
                bbox_vals = boxes[ln]
                bbox = BoundingBox(x0=bbox_vals[0], y0=bbox_vals[1], x1=bbox_vals[2], y1=bbox_vals[3], units="px", page_width=img.size[0], page_height=img.size[1])
                bid = generate_block_id(result.document_id, "paragraph", 1, order, line_str)
                result.content_blocks.append(ContentBlock(
                    block_id=bid,
                    document_id=result.document_id,
                    element_type="paragraph",
                    text=line_str,
                    normalized_text=normalize_text(line_str),
                    page_number=1,
                    reading_order=order,
                    bounding_box=bbox,
                    extraction_method="ocr",
                ))
                order += 1

            figure.ocr_text = "\n".join(all_lines)
            if figure.ocr_text:
                result.document_metadata.ocr_applied = True
                result.document_metadata.ocr_engine = f"Tesseract ({self.options.ocr_languages})"
        except Exception as e:
            err = str(e).lower()
            code = "TESSERACT_BINARY_MISSING" if ("not installed" in err or "not found" in err) else "OCR_FAILED"
            result.extraction_warnings.append(ExtractionWarning(code=code, message=f"OCR warning: {e}"))
