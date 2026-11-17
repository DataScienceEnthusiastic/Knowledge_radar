"""
Demonstration script for multimodal document extraction and downstream storage.

Demonstrates:
1. Extracting digital PDFs (headings, paragraphs, bounding boxes).
2. Extracting standalone images (dimensions, OCR text, figure metadata).
3. Extracting webpage URLs (SSRF safety, DOM hierarchy, tables).
4. Extracting YouTube video URLs (metadata, timestamped transcript segments).
5. Downstream storage into SQLite relational tables and ChromaDB vector collections.
"""

import os
import io
import json
import sqlite3
import pymupdf
from PIL import Image

from modules.extractors import (
    extract_source,
    ExtractionOptions,
    store_in_chromadb,
    store_in_sqlite,
)

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


def create_sample_pdf(filepath: str):
    """Generates a sample synthetic PDF with headings, paragraphs, and a table."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    # Title & Headings
    page.insert_text((50, 60), "1. Executive Summary: AI Ingestion Engine", fontsize=18)
    page.insert_text(
        (50, 100),
        "Knowledge Radar extracts high-fidelity multimodal representations from diverse documents.",
        fontsize=11,
    )
    page.insert_text((50, 150), "1.1 Storage Architecture", fontsize=14)
    page.insert_text(
        (50, 180),
        "Documents are parsed into structured content blocks, tables, and figures with full provenance.",
        fontsize=11,
    )

    # Table
    page.draw_rect(pymupdf.Rect(50, 220, 500, 250), fill=(0.9, 0.9, 0.9), color=(0, 0, 0))
    page.insert_text((60, 240), "Target Store     Data Model       Search Strategy", fontsize=11)
    page.draw_rect(pymupdf.Rect(50, 250, 500, 280), color=(0, 0, 0))
    page.insert_text((60, 270), "ChromaDB         Vector Chunks    Cosine Similarity", fontsize=11)
    page.draw_rect(pymupdf.Rect(50, 280, 500, 310), color=(0, 0, 0))
    page.insert_text((60, 300), "SQLite           Relational Rows  SQL / Exact Match", fontsize=11)

    doc.save(filepath)
    doc.close()


def create_sample_image(filepath: str):
    """Generates a sample PNG image."""
    img = Image.new("RGB", (400, 200), color=(50, 120, 180))
    img.save(filepath, format="PNG")


def main():
    print("=" * 70)
    print("[*] KNOWLEDGE RADAR: MULTIMODAL EXTRACTION DEMONSTRATION")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output")
    os.makedirs(output_dir, exist_ok=True)

    sample_pdf_path = os.path.join(output_dir, "sample_report.pdf")
    sample_img_path = os.path.join(output_dir, "sample_diagram.png")
    sqlite_db_path = os.path.join(output_dir, "knowledge_radar.db")

    create_sample_pdf(sample_pdf_path)
    create_sample_image(sample_img_path)

    # -------------------------------------------------------------------------
    # 1. Extract PDF Document
    # -------------------------------------------------------------------------
    print("\n[1/4] Extracting PDF Document...")
    pdf_result = extract_source(sample_pdf_path)
    print(f"  • Document ID : {pdf_result.document_id}")
    print(f"  • Pages       : {pdf_result.document_metadata.page_count}")
    print(f"  • Blocks      : {len(pdf_result.content_blocks)}")
    print(f"  • Status      : {pdf_result.extraction_status}")
    print("  • Sample Headings Detected:")
    for b in pdf_result.content_blocks:
        if b.element_type == "heading":
            print(f"      - {b.text} (Level path: {b.section_path})")

    # -------------------------------------------------------------------------
    # 2. Extract Standalone Image
    # -------------------------------------------------------------------------
    print("\n[2/4] Extracting Standalone Image...")
    img_result = extract_source(sample_img_path)
    print(f"  • Document ID : {img_result.document_id}")
    print(f"  • Dimensions  : {img_result.figures[0].image_dimensions}")
    print(f"  • Format      : {img_result.figures[0].image_format}")
    print(f"  • Status      : {img_result.extraction_status}")

    # -------------------------------------------------------------------------
    # 3. Extract Webpage URL
    # -------------------------------------------------------------------------
    print("\n[3/4] Extracting Webpage URL (Python.org Docs)...")
    try:
        web_result = extract_source("https://docs.python.org/3/")
        print(f"  • Title       : {web_result.title}")
        print(f"  • Blocks      : {len(web_result.content_blocks)}")
        print(f"  • Links       : {len(web_result.links)}")
        print(f"  • Status      : {web_result.extraction_status}")
    except Exception as e:
        print(f"  • Web extraction skipped/failed: {e}")
        web_result = None

    # -------------------------------------------------------------------------
    # 4. Extract YouTube URL
    # -------------------------------------------------------------------------
    print("\n[4/4] Extracting YouTube Video (Public educational video)...")
    try:
        yt_url = "https://www.youtube.com/watch?v=kCc8FmEb1nY"  # Andrej Karpathy: Intro to Large Language Models
        yt_result = extract_source(yt_url)
        print(f"  • Title       : {yt_result.title}")
        print(f"  • Channel     : {yt_result.document_metadata.channel_name}")
        print(f"  • Segments    : {len(yt_result.transcript_segments)}")
        print(f"  • Status      : {yt_result.extraction_status}")
        if yt_result.transcript_segments:
            seg = yt_result.transcript_segments[0]
            print(f"  • First Timestamp: [{seg.start_time_seconds:.1f}s - {seg.end_time_seconds:.1f}s] '{seg.text}'")
    except Exception as e:
        print(f"  • YouTube extraction note: {e}")
        yt_result = None

    # -------------------------------------------------------------------------
    # 5. Downstream Storage: SQLite Relational Database
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("💾 DOWNSTREAM STORAGE: SQLITE DATABASE")
    print("=" * 70)
    inserted_pdf = store_in_sqlite(pdf_result, sqlite_db_path)
    inserted_img = store_in_sqlite(img_result, sqlite_db_path)
    print(f"  • PDF entities saved to SQLite   : {inserted_pdf}")
    print(f"  • Image entities saved to SQLite : {inserted_img}")

    conn = sqlite3.connect(sqlite_db_path)
    cur = conn.cursor()
    cur.execute("SELECT document_id, source_type, canonical_source FROM documents")
    docs = cur.fetchall()
    print("  • SQLite 'documents' table records:")
    for d in docs:
        print(f"      - ID: {d[0]} | Type: {d[1]} | Source: {d[2]}")
    conn.close()

    # -------------------------------------------------------------------------
    # 6. Downstream Storage: ChromaDB Vector Store
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🔍 DOWNSTREAM STORAGE: CHROMADB VECTOR STORE")
    print("=" * 70)
    if HAS_CHROMA:
        client = chromadb.Client()
        collection = client.get_or_create_collection("knowledge_radar_demo")
        chunks_indexed = store_in_chromadb(pdf_result, collection)
        print(f"  • Indexed {chunks_indexed} structured chunks into ChromaDB collection '{collection.name}'")

        # Query example
        query = "Where are vector chunks and relational rows stored?"
        query_res = collection.query(query_texts=[query], n_results=min(2, chunks_indexed))
        print(f"\n  • Similarity Search for: '{query}':")
        for doc_txt, meta in zip(query_res["documents"][0], query_res["metadatas"][0]):
            print(f"      [Score Match | Type: {meta.get('element_type', 'text')} | Page: {meta.get('page_number', 1)}]")
            snippet = doc_txt.replace("\n", " ")[:120]
            print(f"      Content: \"{snippet}...\"\n")
    else:
        print("  • ChromaDB not installed in environment, skipped live indexing.")

    print("=" * 70)
    print("✅ Extraction and Storage Demonstration Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
