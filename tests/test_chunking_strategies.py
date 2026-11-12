"""
Unit tests for chunking strategies (Sliding Window, Sentence-Aware, Structure-Aware).
"""

import pytest
from modules.parser import chunk_text, chunk_sentences, extract_structure_chunks


def test_chunk_text_basic():
    text = "Knowledge Radar performs dual-context intelligence auditing with local vector retrieval."
    chunks = chunk_text(text, source_name="test_doc", chunk_size=30, chunk_overlap=10)
    assert len(chunks) > 1
    assert all("text" in c and "source" in c and "chunk_index" in c for c in chunks)
    assert chunks[0]["source"] == "test_doc"


def test_chunk_text_boundary_safety():
    # If chunk_overlap >= chunk_size, it must NOT hang in an infinite loop
    text = "Short text to chunk."
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=15)
    assert len(chunks) > 0

    # Empty text
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_sentences_preserves_sentences():
    text = (
        "Quantum computing relies on superposition. "
        "Classical bits can only represent zero or one. "
        "Entanglement links qubits across arbitrary distances."
    )
    chunks = chunk_sentences(text, source_name="quantum_doc", max_chunk_chars=60, overlap_sentences=1)
    assert len(chunks) >= 2

    # Verify complete sentences are preserved without mid-sentence cuts
    for c in chunks:
        # Every chunk should end with a period
        assert c["text"].endswith(".")
        assert "source" in c and c["source"] == "quantum_doc"


def test_chunk_sentences_empty():
    assert chunk_sentences("") == []
    assert chunk_sentences("   ") == []

