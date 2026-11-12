"""
Downstream storage adapters for ChromaDB vector collections and SQLite relational tables.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Union

from modules.extractors.models import ExtractionResult

def store_in_chromadb(
    extraction_result: ExtractionResult,
    collection: Any,
    embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> int:
    """
    Indexes extracted chunks into a ChromaDB collection with rich metadata.
    """
    chunks = extraction_result.to_rag_chunks()
    if not chunks:
        return 0

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = []

    for c in chunks:
        clean_meta = {}
        for k, v in c["metadata"].items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif v is None:
                clean_meta[k] = ""
            else:
                clean_meta[k] = str(v)
        metadatas.append(clean_meta)

    kwargs: Dict[str, Any] = {"ids": ids, "documents": documents, "metadatas": metadatas}
    if embedding_function:
        kwargs["embeddings"] = embedding_function(documents)

    collection.upsert(**kwargs)
    return len(chunks)


def store_in_sqlite(
    extraction_result: ExtractionResult,
    db_path_or_conn: Union[str, sqlite3.Connection],
) -> int:
    """
    Persists structured extraction entities into normalized SQLite relational tables:
    - documents: document metadata & SHA-256 hash
    - content_blocks: paragraphs, headings, reading order, bounding box json
    - tables: structured headers, rows, markdown, and retrieval text
    - figures: captions, dimensions, OCR and vision analysis
    - transcript_segments: timestamped speech intervals
    """
    close_conn = False
    if isinstance(db_path_or_conn, str):
        conn = sqlite3.connect(db_path_or_conn)
        close_conn = True
    else:
        conn = db_path_or_conn

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            source_type TEXT,
            canonical_source TEXT,
            title TEXT,
            file_hash TEXT,
            page_count INTEGER,
            metadata_json TEXT,
            extracted_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS content_blocks (
            block_id TEXT PRIMARY KEY,
            document_id TEXT,
            element_type TEXT,
            text TEXT,
            normalized_text TEXT,
            page_number INTEGER,
            section_path TEXT,
            reading_order INTEGER,
            bounding_box_json TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (document_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tables (
            table_id TEXT PRIMARY KEY,
            document_id TEXT,
            page_number INTEGER,
            caption TEXT,
            column_headers_json TEXT,
            rows_json TEXT,
            markdown_repr TEXT,
            text_repr TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (document_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS figures (
            figure_id TEXT PRIMARY KEY,
            document_id TEXT,
            page_number INTEGER,
            caption TEXT,
            image_reference TEXT,
            visual_description TEXT,
            ocr_text TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (document_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcript_segments (
            segment_id TEXT PRIMARY KEY,
            document_id TEXT,
            start_time_seconds REAL,
            end_time_seconds REAL,
            text TEXT,
            transcript_source TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (document_id)
        )
    """)

    doc = extraction_result
    cur.execute("""
        INSERT OR REPLACE INTO documents 
        (document_id, source_type, canonical_source, title, file_hash, page_count, metadata_json, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        doc.document_id,
        doc.source_type,
        doc.canonical_source,
        doc.title,
        doc.document_metadata.file_hash,
        doc.document_metadata.page_count,
        json.dumps(doc.document_metadata.to_dict()),
        doc.document_metadata.extraction_timestamp_utc,
    ))

    inserted = 1

    for b in doc.content_blocks:
        cur.execute("""
            INSERT OR REPLACE INTO content_blocks
            (block_id, document_id, element_type, text, normalized_text, page_number, section_path, reading_order, bounding_box_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            b.block_id,
            doc.document_id,
            b.element_type,
            b.text,
            b.normalized_text,
            b.page_number,
            json.dumps(b.section_path),
            b.reading_order,
            json.dumps(b.bounding_box.to_dict()) if b.bounding_box else None,
        ))
        inserted += 1

    for t in doc.tables:
        cur.execute("""
            INSERT OR REPLACE INTO tables
            (table_id, document_id, page_number, caption, column_headers_json, rows_json, markdown_repr, text_repr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            t.table_id,
            doc.document_id,
            t.page_number,
            t.caption,
            json.dumps(t.column_headers),
            json.dumps(t.rows),
            t.markdown_representation,
            t.text_representation,
        ))
        inserted += 1

    for f in doc.figures:
        cur.execute("""
            INSERT OR REPLACE INTO figures
            (figure_id, document_id, page_number, caption, image_reference, visual_description, ocr_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            f.figure_id,
            doc.document_id,
            f.page_number,
            f.caption,
            f.image_reference,
            f.visual_description,
            f.ocr_text,
        ))
        inserted += 1

    for s in doc.transcript_segments:
        cur.execute("""
            INSERT OR REPLACE INTO transcript_segments
            (segment_id, document_id, start_time_seconds, end_time_seconds, text, transcript_source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            s.segment_id,
            doc.document_id,
            s.start_time_seconds,
            s.end_time_seconds,
            s.text,
            s.transcript_source,
        ))
        inserted += 1

    conn.commit()
    if close_conn:
        conn.close()

    return inserted
