"""
Unit tests validating benchmark dataset integrity, schemas, doc references, and split isolation.
"""

import os
import json
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "evaluation", "dataset")
DOCS_DIR = os.path.join(DATASET_DIR, "documents")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")


def test_documents_corpus_size():
    doc_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
    assert len(doc_files) >= 50, f"Expected at least 50 documents, got {len(doc_files)}"


def test_queries_and_ground_truth_alignment():
    q_file = os.path.join(DATASET_DIR, "questions.jsonl")
    gt_file = os.path.join(DATASET_DIR, "ground_truth.jsonl")

    assert os.path.exists(q_file)
    assert os.path.exists(gt_file)

    with open(q_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth = [json.loads(line) for line in f if line.strip()]

    assert len(questions) == 120
    assert len(ground_truth) == 120

    q_ids = [q["id"] for q in questions]
    gt_ids = [gt["id"] for gt in ground_truth]
    assert q_ids == gt_ids, "Question IDs must match Ground Truth IDs exactly"

    # Check document existence for all relevant_docs
    existing_docs = set(os.listdir(DOCS_DIR))
    for gt in ground_truth:
        for doc_name in gt["relevant_docs"]:
            assert doc_name in existing_docs, f"Referenced doc '{doc_name}' for query '{gt['id']}' not in documents/"


def test_splits_isolation_and_partitioning():
    val_file = os.path.join(SPLITS_DIR, "validation.jsonl")
    test_file = os.path.join(SPLITS_DIR, "test.jsonl")

    assert os.path.exists(val_file)
    assert os.path.exists(test_file)

    with open(val_file, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f if line.strip()]

    with open(test_file, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]

    assert len(val_samples) == 50, f"Validation set must contain exactly 50 queries, got {len(val_samples)}"
    assert len(test_samples) == 70, f"Test set must contain exactly 70 queries, got {len(test_samples)}"

    val_ids = set(s["id"] for s in val_samples)
    test_ids = set(s["id"] for s in test_samples)

    # Disjoint check
    intersection = val_ids.intersection(test_ids)
    assert len(intersection) == 0, f"Data leakage! Validation and Test sets share queries: {intersection}"

    # Partition check
    all_ids = val_ids.union(test_ids)
    assert len(all_ids) == 120, "Combined splits must cover all 120 queries"
