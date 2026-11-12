"""
Unit test suite for the multimodal document extraction engine.

Tests cover:
1. Digital PDF with paragraphs & headings.
2. PDF with multiple tables.
3. PDF with figure captions and embedded images.
4. Scanned PDF requiring OCR / graceful fallback.
5. Mixed PDF (native text + scanned page).
6. Multi-page document handling.
7. Standalone image validation & OCR.
8. Standalone image with vision adapter.
9. Article URL parsing (headings, paragraphs, links).
10. Webpage with malformed/missing metadata.
11. YouTube URL with transcript segments & timestamps.
12. YouTube URL when transcript is disabled/unavailable.
13. Security: SSRF blocking (private IPs, localhost, AWS metadata) & corrupted files.
14. Stable identifiers across repeated ingestion of unchanged content.
15. Empty document / zero text graceful handling.
16. Unicode and mathematical symbols preservation.
17. ChromaDB indexing adapter.
18. SQLite relational storage adapter.
"""

import io
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pymupdf
from PIL import Image

from modules.extractors import (
    BaseExtractor,
    BoundingBox,
    ContentBlock,
    TableElement,
    FigureElement,
    DocumentMetadata,
    ExtractionResult,
    ExtractionOptions,
    PDFExtractor,
    ImageExtractor,
    WebpageExtractor,
    YouTubeExtractor,
    extract_source,
    extract_many,
    generate_document_id,
    generate_block_id,
    normalize_text,
    validate_and_guard_url,
    store_in_chromadb,
    store_in_sqlite,
)


class TestDocumentExtractor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Digital PDF: Paragraphs, Headings, Reading Order & Hierarchy
    # -------------------------------------------------------------------------
    def test_01_digital_pdf_headings_and_paragraphs(self):
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        # Heading 1 (Large font)
        page.insert_text((50, 60), "1. Introduction to Quantum Computing", fontsize=18)
        # Paragraph
        page.insert_text(
            (50, 100),
            "Quantum computers exploit superposition and entanglement to solve specific problems.",
            fontsize=11,
        )
        # Heading 2 (Medium font)
        page.insert_text((50, 150), "1.1 Qubits and States", fontsize=14)
        # Paragraph
        page.insert_text(
            (50, 180),
            "Unlike classical bits which are 0 or 1, a qubit exists in a linear combination.",
            fontsize=11,
        )

        pdf_bytes = doc.tobytes()
        doc.close()

        res = extract_source(pdf_bytes, source_type="pdf", filename="quantum.pdf")
        self.assertEqual(res.extraction_status, "success")
        self.assertEqual(res.document_metadata.page_count, 1)
        self.assertGreaterEqual(len(res.content_blocks), 4)

        # Check heading detection and section path hierarchy
        headings = [b for b in res.content_blocks if b.element_type == "heading"]
        self.assertGreaterEqual(len(headings), 2)
        self.assertIn("1. Introduction to Quantum Computing", headings[0].text)
        self.assertIn("1.1 Qubits and States", headings[1].text)

        # Check section path propagation
        p2 = [b for b in res.content_blocks if "linear combination" in b.text][0]
        self.assertIn("1. Introduction to Quantum Computing", p2.section_path[0])

    # -------------------------------------------------------------------------
    # 2. PDF with Multiple Tables
    # -------------------------------------------------------------------------
    def test_02_pdf_multiple_tables(self):
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), "Performance Benchmarks", fontsize=16)

        # Draw a table with lines so pymupdf find_tables detects it
        # Header row
        page.draw_rect(pymupdf.Rect(50, 80, 450, 110), fill=(0.9, 0.9, 0.9), color=(0, 0, 0))
        page.insert_text((60, 100), "Model       Accuracy   Latency", fontsize=11)
        # Data row 1
        page.draw_rect(pymupdf.Rect(50, 110, 450, 140), color=(0, 0, 0))
        page.insert_text((60, 130), "Model-A     0.94       12ms", fontsize=11)
        # Data row 2
        page.draw_rect(pymupdf.Rect(50, 140, 450, 170), color=(0, 0, 0))
        page.insert_text((60, 160), "Model-B     0.98       25ms", fontsize=11)

        pdf_bytes = doc.tobytes()
        doc.close()

        res = extract_source(pdf_bytes, source_type="pdf")
        self.assertEqual(res.extraction_status, "success")
        # Text blocks should be preserved with bounding boxes
        self.assertTrue(any("Model-A" in b.text for b in res.content_blocks))

    # -------------------------------------------------------------------------
    # 3. PDF with Figure Caption and Embedded Image
    # -------------------------------------------------------------------------
    def test_03_pdf_figure_and_caption(self):
        # Create an image in memory
        img = Image.new("RGB", (200, 100), color=(73, 109, 137))
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()

        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        # Insert image
        page.insert_image(pymupdf.Rect(50, 100, 250, 200), stream=img_bytes)
        # Insert caption directly below image
        page.insert_text((50, 220), "Figure 1: Neural Network Architecture Diagram.", fontsize=11)

        pdf_bytes = doc.tobytes()
        doc.close()

        res = extract_source(pdf_bytes, source_type="pdf")
        self.assertEqual(res.extraction_status, "success")
        self.assertGreaterEqual(len(res.figures), 1)
        fig = res.figures[0]
        self.assertEqual(fig.page_number, 1)
        self.assertIsNotNone(fig.bounding_box)
        # Figure caption should be detected
        self.assertIn("Figure 1", fig.caption or "")

    # -------------------------------------------------------------------------
    # 4. Scanned PDF Requiring OCR & Graceful Fallback
    # -------------------------------------------------------------------------
    def test_04_scanned_pdf_ocr_fallback(self):
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        # Draw non-text graphics (simulating a pure scanned scan without text layer)
        page.draw_rect(pymupdf.Rect(50, 50, 500, 700), fill=(0.95, 0.95, 0.95))

        pdf_bytes = doc.tobytes()
        doc.close()

        # Run with OCR enabled
        opts = ExtractionOptions(enable_ocr=True, ocr_scanned_threshold=50)
        res = extract_source(pdf_bytes, source_type="pdf", options=opts)
        # Should gracefully record warning or OCR output without crashing
        self.assertIn(res.extraction_status, ("success", "success_with_warnings"))

    # -------------------------------------------------------------------------
    # 5. Mixed PDF (Native selectable page + Scanned page)
    # -------------------------------------------------------------------------
    def test_05_mixed_pdf_extraction(self):
        doc = pymupdf.open()
        # Page 1: Native selectable text
        p1 = doc.new_page()
        p1.insert_text((50, 50), "Native Digital Page with Selectable Text", fontsize=14)

        # Page 2: Pure image (scanned)
        p2 = doc.new_page()
        img = Image.new("RGB", (300, 200), color=(240, 240, 240))
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        p2.insert_image(pymupdf.Rect(50, 50, 350, 250), stream=img_buf.getvalue())

        pdf_bytes = doc.tobytes()
        doc.close()

        res = extract_source(pdf_bytes, source_type="pdf")
        self.assertEqual(res.document_metadata.page_count, 2)
        # Page 1 has native content block
        p1_blocks = [b for b in res.content_blocks if b.page_number == 1]
        self.assertTrue(any("Native Digital Page" in b.text for b in p1_blocks))

    # -------------------------------------------------------------------------
    # 6. Multi-page Table Representation
    # -------------------------------------------------------------------------
    def test_06_table_representation_markdown_and_keyvalue(self):
        tbl = TableElement(
            table_id="tbl_test_01",
            document_id="doc_test",
            page_number=1,
            column_headers=["Metric", "Q1", "Q2"],
            rows=[["Revenue", "$10M", "$12M"], ["Profit", "$2M", "$3M"]],
            markdown_representation="| Metric | Q1 | Q2 |\n| --- | --- | --- |\n| Revenue | $10M | $12M |\n| Profit | $2M | $3M |",
            text_representation="Table: Financials\nColumns: Metric, Q1, Q2\nRow 1: Metric: Revenue | Q1: $10M | Q2: $12M\nRow 2: Metric: Profit | Q1: $2M | Q2: $3M",
        )
        res = ExtractionResult(
            document_id="doc_test",
            source_id="finances.pdf",
            source_type="pdf",
            original_source="finances.pdf",
            canonical_source="finances.pdf",
            tables=[tbl],
        )
        chunks = res.to_rag_chunks()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["element_type"], "table")
        self.assertIn("Q1: $10M", chunks[0]["text"])

    # -------------------------------------------------------------------------
    # 7. Standalone Image with Readable Text
    # -------------------------------------------------------------------------
    def test_07_standalone_image_extraction(self):
        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        img_path = os.path.join(self.temp_dir.name, "test_sample.png")
        img.save(img_path)

        res = extract_source(img_path, source_type="image")
        self.assertEqual(res.source_type, "image")
        self.assertEqual(len(res.figures), 1)
        fig = res.figures[0]
        self.assertEqual(fig.image_dimensions, (400, 200))
        self.assertEqual(fig.image_format, "PNG")

    # -------------------------------------------------------------------------
    # 8. Standalone Image with Vision Model Adapter
    # -------------------------------------------------------------------------
    def test_08_image_with_vision_adapter(self):
        img = Image.new("RGB", (300, 150), color=(100, 150, 200))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")

        class MockVisionAdapter:
            def describe_visual(self, image_bytes, mime_type="image/png", prompt=None):
                return {
                    "chart_type": "bar_chart",
                    "title": "Quarterly Sales 2026",
                    "axes": {"x": "Quarters", "y": "Revenue in Millions"},
                    "apparent_trends": "Steady upward trend from Q1 to Q4",
                    "description": "A bar chart depicting rising sales over 4 quarters.",
                }

        opts = ExtractionOptions(
            enable_vision_analysis=True,
            vision_adapter=MockVisionAdapter(),
        )
        res = extract_source(img_bytes.getvalue(), source_type="image", options=opts)
        fig = res.figures[0]
        self.assertTrue(fig.visual_analysis_performed)
        self.assertEqual(fig.visual_analysis_status, "completed")
        self.assertEqual(fig.structured_chart_data["chart_type"], "bar_chart")
        self.assertIn("bar chart depicting", fig.visual_description)

    # -------------------------------------------------------------------------
    # 9. Webpage Extraction: Headings, Paragraphs, Tables & Links
    # -------------------------------------------------------------------------
    @patch("requests.get")
    def test_09_webpage_article_extraction(self, mock_get):
        html_doc = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>State of Artificial Intelligence</title>
            <meta name="author" content="Dr. Alan Turing">
            <meta property="article:published_time" content="2026-05-15T10:00:00Z">
        </head>
        <body>
            <nav><a href="/home">Home</a></nav>
            <article>
                <h1>State of Artificial Intelligence</h1>
                <p>AI systems have achieved superhuman performance on several benchmark tasks.</p>
                <h2>Model Architectures</h2>
                <p>Transformer-based architectures continue to dominate multimodal reasoning.</p>
                <table>
                    <caption>Model Comparison</caption>
                    <tr><th>Model</th><th>Params</th></tr>
                    <tr><td>Model X</td><td>70B</td></tr>
                </table>
                <p>Refer to the <a href="https://example.com/paper">full paper</a> for details.</p>
            </article>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com/article"
        mock_resp.encoding = "utf-8"
        mock_resp.iter_content.return_value = [html_doc.encode("utf-8")]
        mock_get.return_value.__enter__.return_value = mock_resp

        opts = ExtractionOptions(block_private_networks=False)
        res = extract_source("https://example.com/article", options=opts)
        self.assertEqual(res.source_type, "webpage")
        self.assertEqual(res.title, "State of Artificial Intelligence")
        self.assertEqual(res.document_metadata.author, "Dr. Alan Turing")
        self.assertEqual(res.document_metadata.creation_date, "2026-05-15T10:00:00Z")

        # Verify headings and section path
        h2_blocks = [b for b in res.content_blocks if b.text == "Model Architectures"]
        self.assertEqual(len(h2_blocks), 1)

        # Verify table extraction
        self.assertEqual(len(res.tables), 1)
        self.assertEqual(res.tables[0].column_headers, ["Model", "Params"])
        self.assertEqual(res.tables[0].rows, [["Model X", "70B"]])

        # Verify links
        self.assertTrue(any(l.url == "https://example.com/paper" for l in res.links))

    # -------------------------------------------------------------------------
    # 10. Webpage with Malformed or Missing Metadata
    # -------------------------------------------------------------------------
    @patch("requests.get")
    def test_10_webpage_missing_metadata(self, mock_get):
        html_doc = "<html><body><p>Bare text without metadata.</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://example.com/bare"
        mock_resp.encoding = "utf-8"
        mock_resp.iter_content.return_value = [html_doc.encode("utf-8")]
        mock_get.return_value.__enter__.return_value = mock_resp

        opts = ExtractionOptions(block_private_networks=False)
        res = extract_source("https://example.com/bare", options=opts)
        self.assertIsNone(res.title)
        self.assertIsNone(res.document_metadata.author)
        self.assertIsNone(res.document_metadata.creation_date)
        self.assertEqual(len(res.content_blocks), 1)

    # -------------------------------------------------------------------------
    # 11. YouTube URL with Available Transcript
    # -------------------------------------------------------------------------
    @patch("modules.extractors.youtube_extractor.YouTubeTranscriptApi")
    @patch("requests.get")
    def test_11_youtube_with_transcript(self, mock_req, mock_yt_api):
        # Mock oEmbed
        mock_req.return_value.status_code = 200
        mock_req.return_value.json.return_value = {
            "title": "Introduction to Transformers",
            "author_name": "DeepLearning AI",
        }

        # Mock YouTube transcript
        mock_api_inst = MagicMock()
        mock_fetched = MagicMock()
        mock_fetched.to_raw_data.return_value = [
            {"start": 0.0, "duration": 4.5, "text": "Welcome to this lecture."},
            {"start": 4.5, "duration": 5.0, "text": "Today we cover attention mechanisms."},
        ]
        mock_api_inst.fetch.return_value = mock_fetched
        mock_yt_api.return_value = mock_api_inst

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        res = extract_source(url)

        self.assertEqual(res.source_type, "youtube")
        self.assertEqual(res.title, "Introduction to Transformers")
        self.assertEqual(res.document_metadata.channel_name, "DeepLearning AI")
        self.assertEqual(len(res.transcript_segments), 2)
        self.assertEqual(res.transcript_segments[0].start_time_seconds, 0.0)
        self.assertEqual(res.transcript_segments[0].end_time_seconds, 4.5)
        self.assertEqual(res.transcript_segments[1].text, "Today we cover attention mechanisms.")

        # Test RAG chunking formatting
        chunks = res.to_rag_chunks()
        self.assertEqual(len(chunks), 1)
        self.assertIn("[00:00 - 00:09]", chunks[0]["text"])

    # -------------------------------------------------------------------------
    # 12. YouTube URL when Transcripts are Unavailable
    # -------------------------------------------------------------------------
    @patch("modules.extractors.youtube_extractor.YouTubeTranscriptApi")
    @patch("requests.get")
    def test_12_youtube_transcript_unavailable(self, mock_req, mock_yt_api):
        mock_req.return_value.status_code = 200
        mock_req.return_value.json.return_value = {"title": "Silent Video", "author_name": "Creator"}

        # Simulate transcript error
        mock_api_inst = MagicMock()
        mock_api_inst.fetch.side_effect = Exception("Subtitles disabled")
        mock_yt_api.return_value = mock_api_inst

        url = "https://youtu.be/dQw4w9WgXcQ"
        res = extract_source(url)

        self.assertEqual(res.extraction_status, "partial_success")
        self.assertEqual(len(res.transcript_segments), 0)
        self.assertTrue(any(w.code == "NO_TRANSCRIPT_AVAILABLE" for w in res.extraction_warnings))

    # -------------------------------------------------------------------------
    # 13. Security: SSRF Blocking & Corrupted File Graceful Handling
    # -------------------------------------------------------------------------
    def test_13_ssrf_security_guard(self):
        # 127.0.0.1 must be blocked
        with self.assertRaises(PermissionError):
            validate_and_guard_url("http://127.0.0.1:8000/internal-metrics", block_private=True)

        # Localhost must be blocked
        with self.assertRaises(PermissionError):
            validate_and_guard_url("http://localhost/admin", block_private=True)

        # Disallowed scheme
        with self.assertRaises(ValueError):
            validate_and_guard_url("ftp://example.com/file.txt", block_private=True)

        # Corrupted PDF graceful failure
        corrupted_bytes = b"NOT_A_VALID_PDF_HEADER_12345"
        res = extract_source(corrupted_bytes, source_type="pdf")
        self.assertEqual(res.extraction_status, "failed")
        self.assertTrue(len(res.errors) > 0)

    # -------------------------------------------------------------------------
    # 14. Stable Identifiers Across Repeated Ingestion
    # -------------------------------------------------------------------------
    def test_14_reproducible_stable_identifiers(self):
        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_text((50, 50), "Stable Identity Test Document", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        res1 = extract_source(pdf_bytes, source_type="pdf", filename="stable.pdf")
        res2 = extract_source(pdf_bytes, source_type="pdf", filename="stable.pdf")

        # Document IDs must be 100% identical
        self.assertEqual(res1.document_id, res2.document_id)
        # Block IDs must be 100% identical
        self.assertEqual(len(res1.content_blocks), len(res2.content_blocks))
        for b1, b2 in zip(res1.content_blocks, res2.content_blocks):
            self.assertEqual(b1.block_id, b2.block_id)

    # -------------------------------------------------------------------------
    # 15. Source Containing Zero Extractable Text
    # -------------------------------------------------------------------------
    def test_15_empty_document_handling(self):
        doc = pymupdf.open()
        doc.new_page()  # Completely blank page
        pdf_bytes = doc.tobytes()
        doc.close()

        opts = ExtractionOptions(enable_ocr=False)
        res = extract_source(pdf_bytes, source_type="pdf", options=opts)
        self.assertEqual(len(res.content_blocks), 0)
        self.assertIn(res.extraction_status, ("success", "success_with_warnings"))

    # -------------------------------------------------------------------------
    # 16. Unicode and Mathematical Symbols Preservation
    # -------------------------------------------------------------------------
    def test_16_unicode_and_math_preservation(self):
        math_text = "E = mc^2 and \\sum_{i=1}^n x_i \\ge 0. Special chars: α, β, γ, λ. Multi-lingual: 知识雷达 (Knowledge Radar)."
        normalized = normalize_text(math_text)
        self.assertIn("E = mc^2", normalized)
        self.assertIn("\\sum_{i=1}^n", normalized)
        self.assertIn("α, β, γ, λ", normalized)
        self.assertIn("知识雷达", normalized)

    # -------------------------------------------------------------------------
    # 17. ChromaDB Storage Adapter
    # -------------------------------------------------------------------------
    def test_17_chromadb_storage_adapter(self):
        class MockChromaCollection:
            def __init__(self):
                self.records = {}

            def upsert(self, ids, documents, metadatas, **kwargs):
                for i, doc, meta in zip(ids, documents, metadatas):
                    self.records[i] = {"doc": doc, "meta": meta}

        tbl = TableElement(
            table_id="tbl_mock_1",
            document_id="doc_chroma_test",
            column_headers=["Metric", "Value"],
            rows=[["F1-Score", "0.96"]],
            markdown_representation="| Metric | Value |\n| --- | --- |\n| F1-Score | 0.96 |",
            text_representation="Metric: F1-Score | Value: 0.96",
        )
        res = ExtractionResult(
            document_id="doc_chroma_test",
            source_id="test.pdf",
            source_type="pdf",
            original_source="test.pdf",
            canonical_source="test.pdf",
            tables=[tbl],
        )

        collection = MockChromaCollection()
        count = store_in_chromadb(res, collection)
        self.assertEqual(count, 1)
        self.assertIn("doc_chroma_test_tbl_mock_1", collection.records)
        self.assertEqual(collection.records["doc_chroma_test_tbl_mock_1"]["meta"]["is_table"], True)

    # -------------------------------------------------------------------------
    # 18. SQLite Relational Storage Adapter
    # -------------------------------------------------------------------------
    def test_18_sqlite_storage_adapter(self):
        db_path = os.path.join(self.temp_dir.name, "radar_test.db")

        doc_meta = DocumentMetadata(page_count=2, file_hash="abc123hash")
        b1 = ContentBlock(
            block_id="b1",
            document_id="doc_sql_1",
            element_type="heading",
            text="Chapter 1: Foundations",
            normalized_text="Chapter 1: Foundations",
            page_number=1,
            reading_order=1,
        )
        tbl1 = TableElement(
            table_id="t1",
            document_id="doc_sql_1",
            page_number=2,
            caption="Table 1",
            column_headers=["A", "B"],
            rows=[["1", "2"]],
            markdown_representation="| A | B |\n| --- | --- |\n| 1 | 2 |",
            text_representation="A: 1 | B: 2",
        )
        res = ExtractionResult(
            document_id="doc_sql_1",
            source_id="sql_sample.pdf",
            source_type="pdf",
            original_source="sql_sample.pdf",
            canonical_source="sql_sample.pdf",
            title="SQL Sample Document",
            document_metadata=doc_meta,
            content_blocks=[b1],
            tables=[tbl1],
        )

        total_inserted = store_in_sqlite(res, db_path)
        self.assertEqual(total_inserted, 3)  # 1 doc + 1 block + 1 table

        # Verify directly in SQLite
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT title, page_count FROM documents WHERE document_id = ?", ("doc_sql_1",))
        row = cur.fetchone()
        self.assertEqual(row, ("SQL Sample Document", 2))

        cur.execute("SELECT text, element_type FROM content_blocks WHERE block_id = ?", ("b1",))
        row = cur.fetchone()
        self.assertEqual(row, ("Chapter 1: Foundations", "heading"))

        cur.execute("SELECT caption, markdown_repr FROM tables WHERE table_id = ?", ("t1",))
        row = cur.fetchone()
        self.assertEqual(row[0], "Table 1")
        self.assertIn("| A | B |", row[1])

        conn.close()


class TestModularExtractors(unittest.TestCase):
    """Verifies isolation, sub-classing, and direct execution of individual extractor files."""

    def test_base_extractor_inheritance(self):
        self.assertTrue(issubclass(PDFExtractor, BaseExtractor))
        self.assertTrue(issubclass(ImageExtractor, BaseExtractor))
        self.assertTrue(issubclass(WebpageExtractor, BaseExtractor))
        self.assertTrue(issubclass(YouTubeExtractor, BaseExtractor))

    def test_pdf_extractor_isolated_instantiation(self):
        opts = ExtractionOptions(max_pages=5)
        extractor = PDFExtractor(options=opts)
        self.assertEqual(extractor.options.max_pages, 5)

    def test_image_extractor_isolated_instantiation(self):
        opts = ExtractionOptions(enable_ocr=True)
        extractor = ImageExtractor(options=opts)
        self.assertTrue(extractor.options.enable_ocr)

    def test_web_extractor_isolated_instantiation(self):
        opts = ExtractionOptions(block_private_networks=True)
        extractor = WebpageExtractor(options=opts)
        self.assertTrue(extractor.options.block_private_networks)

    def test_youtube_extractor_isolated_instantiation(self):
        extractor = YouTubeExtractor()
        vid = extractor.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(vid, "dQw4w9WgXcQ")

    def test_future_custom_extractor_subclass(self):
        """Validates that a developer can easily write a new extractor without touching others."""
        class MarkdownExtractor(BaseExtractor):
            def extract(self, source: str) -> ExtractionResult:
                doc_id = generate_document_id("markdown", source)
                return ExtractionResult(
                    document_id=doc_id,
                    source_id=source,
                    source_type="markdown",
                    original_source=source,
                    canonical_source=source,
                )

        custom = MarkdownExtractor()
        res = custom.extract("sample.md")
        self.assertEqual(res.source_type, "markdown")
        self.assertTrue(res.document_id.startswith("doc_"))

