"""
Webpage article extraction engine.
Handles SSRF validation, content streaming, DOM hierarchy, and tables.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB_DEPS = True
except ImportError:
    requests = None
    BeautifulSoup = None
    HAS_WEB_DEPS = False

from modules.extractors.base import BaseExtractor
from modules.extractors.models import (
    ContentBlock,
    DocumentMetadata,
    ExtractionError,
    ExtractionOptions,
    ExtractionResult,
    ExtractionStatistics,
    ExtractionWarning,
    FigureElement,
    LinkElement,
    TableElement,
)
from modules.extractors.utils import generate_block_id, generate_document_id, normalize_text, validate_and_guard_url

logger = logging.getLogger("document_extractor.web")

class WebpageExtractor(BaseExtractor):
    """
    Safely fetches and parses web articles.
    Enforces SSRF prevention, streaming size boundaries, and semantic DOM extraction.
    """

    def __init__(self, options: Optional[ExtractionOptions] = None):
        self.options = options or ExtractionOptions()

    def extract(self, url: str) -> ExtractionResult:
        start_time = time.time()
        try:
            safe_url = validate_and_guard_url(url, block_private=self.options.block_private_networks)
        except Exception as e:
            res = ExtractionResult(
                document_id=generate_document_id("webpage", url),
                source_id=url,
                source_type="webpage",
                original_source=url,
                canonical_source=url,
            )
            res.errors.append(ExtractionError(code="SSRF_SECURITY_REJECTION", message=f"URL rejected: {e}"))
            res.extraction_status = "failed"
            return res

        canonical = safe_url
        doc_id = generate_document_id("webpage", canonical)

        metadata = DocumentMetadata(
            source_url=url,
            canonical_url=canonical,
            mime_type="text/html",
        )

        result = ExtractionResult(
            document_id=doc_id,
            source_id=canonical,
            source_type="webpage",
            original_source=url,
            canonical_source=canonical,
            document_metadata=metadata,
        )

        if not HAS_WEB_DEPS:
            result.errors.append(ExtractionError(code="NO_WEB_DEPS", message="requests and beautifulsoup4 required."))
            result.extraction_status = "failed"
            return res

        try:
            headers = {"User-Agent": self.options.user_agent}
            with requests.get(
                canonical, headers=headers, timeout=self.options.request_timeout_seconds, stream=True, allow_redirects=True
            ) as resp:
                resp.raise_for_status()

                # Re-verify redirected target
                if resp.url != canonical:
                    validate_and_guard_url(resp.url, block_private=self.options.block_private_networks)
                    canonical = resp.url
                    result.canonical_source = canonical
                    result.document_metadata.canonical_url = canonical

                chunks = []
                total = 0
                for chk in resp.iter_content(chunk_size=65536):
                    total += len(chk)
                    if total > self.options.max_file_size_bytes:
                        raise ValueError(f"Response exceeded size limit {self.options.max_file_size_bytes} bytes.")
                    chunks.append(chk)

                raw_bytes = b"".join(chunks)
                html = raw_bytes.decode(resp.encoding or "utf-8", errors="replace")
                result.document_metadata.file_size_bytes = len(raw_bytes)
                result.document_metadata.file_hash = hashlib.sha256(raw_bytes).hexdigest()

        except Exception as e:
            result.errors.append(ExtractionError(code="HTTP_FETCH_ERROR", message=f"Fetch failed: {e}"))
            result.extraction_status = "failed"
            return result

        self._parse_dom(html, canonical, result)

        result.extraction_statistics = ExtractionStatistics(
            page_count=1,
            content_block_count=len(result.content_blocks),
            table_count=len(result.tables),
            figure_count=len(result.figures),
            link_count=len(result.links),
            extraction_duration_seconds=round(time.time() - start_time, 3),
            warning_count=len(result.extraction_warnings),
            error_count=len(result.errors),
        )

        result.extraction_status = "failed" if result.errors else (
            "success_with_warnings" if result.extraction_warnings else "success"
        )
        return result

    def _parse_dom(self, html: str, base_url: str, result: ExtractionResult):
        soup = BeautifulSoup(html, "html.parser")

        # Title
        t_tag = soup.find("title") or soup.find("meta", property="og:title")
        if t_tag:
            result.title = t_tag.get("content") if t_tag.name == "meta" else t_tag.get_text().strip()
            result.document_metadata.title = result.title

        # Metadata
        author = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", property="article:author")
        if author:
            result.document_metadata.author = author.get("content", "").strip() or None

        date = soup.find("meta", property="article:published_time") or soup.find("meta", attrs={"name": "publication_date"})
        if date:
            result.document_metadata.creation_date = date.get("content", "").strip() or None

        # Clean noise tags
        for el in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            el.decompose()

        container = soup.find("article") or soup.find("main") or soup.body or soup
        section_stack: List[str] = []
        order = 1

        for el in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "figure", "img", "a"]):
            tag = el.name.lower()

            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                txt = el.get_text().strip()
                if not txt:
                    continue
                level = int(tag[1])
                if level == 1:
                    section_stack = [txt]
                elif len(section_stack) >= level:
                    section_stack = section_stack[:level - 1] + [txt]
                else:
                    section_stack.append(txt)

                bid = generate_block_id(result.document_id, "heading", 1, order, txt)
                result.content_blocks.append(ContentBlock(
                    block_id=bid,
                    document_id=result.document_id,
                    element_type="heading",
                    text=txt,
                    normalized_text=normalize_text(txt),
                    page_number=1,
                    section_path=list(section_stack),
                    reading_order=order,
                    extraction_method="html_dom",
                    metadata={"heading_level": level},
                ))
                order += 1

            elif tag == "p":
                txt = el.get_text().strip()
                if not txt or len(txt) < 3:
                    continue
                bid = generate_block_id(result.document_id, "paragraph", 1, order, txt)
                result.content_blocks.append(ContentBlock(
                    block_id=bid,
                    document_id=result.document_id,
                    element_type="paragraph",
                    text=txt,
                    normalized_text=normalize_text(txt),
                    page_number=1,
                    section_path=list(section_stack),
                    reading_order=order,
                    extraction_method="html_dom",
                ))
                order += 1

            elif tag in ("ul", "ol"):
                for li in el.find_all("li", recursive=False):
                    txt = li.get_text().strip()
                    if txt:
                        bid = generate_block_id(result.document_id, "list", 1, order, txt)
                        result.content_blocks.append(ContentBlock(
                            block_id=bid,
                            document_id=result.document_id,
                            element_type="list",
                            text=txt,
                            normalized_text=normalize_text(txt),
                            page_number=1,
                            section_path=list(section_stack),
                            reading_order=order,
                            extraction_method="html_dom",
                        ))
                        order += 1

            elif tag == "table" and self.options.extract_tables:
                self._parse_html_table(el, result, section_stack, order)
                order += 1

            elif tag in ("figure", "img") and self.options.extract_figures:
                img_tag = el if tag == "img" else el.find("img")
                if img_tag and img_tag.get("src"):
                    src = urljoin(base_url, img_tag.get("src"))
                    alt = img_tag.get("alt", "").strip() or None
                    caption_el = el.find("figcaption") if tag == "figure" else None
                    caption = caption_el.get_text().strip() if caption_el else alt
                    fig_id = f"fig_{result.document_id[:8]}_1_{len(result.figures)+1}"
                    result.figures.append(FigureElement(
                        figure_id=fig_id,
                        document_id=result.document_id,
                        page_number=1,
                        caption=caption,
                        caption_is_inferred=bool(alt and not caption_el),
                        section_heading=section_stack[-1] if section_stack else None,
                        image_reference=src,
                        extraction_method="html_image",
                    ))

            elif tag == "a" and self.options.extract_links:
                href = el.get("href")
                anchor = el.get_text().strip()
                if href and anchor and not href.startswith(("#", "javascript:", "mailto:")):
                    full_url = urljoin(base_url, href)
                    lid = f"lnk_{result.document_id[:8]}_1_{len(result.links)+1}"
                    result.links.append(LinkElement(
                        link_id=lid,
                        document_id=result.document_id,
                        url=full_url,
                        text=anchor,
                        page_number=1,
                        section_heading=section_stack[-1] if section_stack else None,
                        is_external=urlparse(full_url).netloc != urlparse(base_url).netloc,
                    ))

    def _parse_html_table(self, tbl_tag: Any, result: ExtractionResult, section_stack: List[str], order: int):
        headers, rows = [], []
        caption_tag = tbl_tag.find("caption")
        caption = caption_tag.get_text().strip() if caption_tag else None

        for tr in tbl_tag.find_all("tr"):
            th = tr.find_all("th")
            if th and not headers:
                headers = [c.get_text().strip() for c in th]
            else:
                td = tr.find_all(["td", "th"])
                if td:
                    rows.append([c.get_text().strip() for c in td])

        if not headers and rows:
            headers = [f"Col_{i+1}" for i in range(len(rows[0]))]
        if not rows and not headers:
            return

        md = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
        md += "\n".join("| " + " | ".join(r) + " |" for r in rows)

        lines = [f"Table: {caption or 'Web Table'}", f"Columns: {', '.join(headers)}"]
        for idx, r in enumerate(rows, 1):
            pairs = [f"{headers[j]}: {c}" for j, c in enumerate(r) if j < len(headers)]
            lines.append(f"Row {idx}: {' | '.join(pairs)}")
        text_repr = "\n".join(lines)

        tbl_id = f"tbl_{result.document_id[:8]}_1_{len(result.tables)+1}"
        result.tables.append(TableElement(
            table_id=tbl_id,
            document_id=result.document_id,
            page_number=1,
            caption=caption,
            section_heading=section_stack[-1] if section_stack else None,
            column_headers=headers,
            rows=rows,
            row_count=len(rows),
            col_count=len(headers),
            extraction_method="html_table",
            markdown_representation=md,
            text_representation=text_repr,
        ))

        result.content_blocks.append(ContentBlock(
            block_id=tbl_id,
            document_id=result.document_id,
            element_type="table",
            text=md,
            normalized_text=normalize_text(text_repr),
            page_number=1,
            section_path=list(section_stack),
            reading_order=order,
            extraction_method="html_table",
        ))
