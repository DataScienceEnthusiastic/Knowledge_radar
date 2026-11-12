# Knowledge Radar Benchmark Dataset (v1.0.0)

## Overview & Governance
- **Total Corpus Size**: 60 Documents (15 Core Diagnostic Documents + 45 Distractor Documents).
- **Total Benchmark Queries**: 120 Curated Queries across 7 Distinct Categories.
- **Ground Truth Validation Status**: 100% `MANUALLY_VERIFIED` with strict relevance bindings and entity checks.

## Partitioning & Test Set Lock
- **Validation Split (`splits/validation.jsonl`)**: 50 Queries used for exploratory parameter tuning and baseline selection.
- **Held-Out Test Split (`splits/test.jsonl`)**: 70 Queries RESERVED for final evaluation.
> **Lock Protocol**: The final test set is evaluated only after all design decisions are frozen. No tuning or configuration selection is performed on test data.

## Query Category Distribution
| Category | Validation (N=50) | Test (N=70) | Total (N=120) | Evaluation Focus |
| :--- | :---: | :---: | :---: | :--- |
| **factual** | 8 | 12 | 20 | Precision@K, exact facts |
| **keyword-heavy** | 8 | 12 | 20 | Lexical BM25 vs Dense vector |
| **semantic** | 8 | 12 | 20 | Paraphrasing & concept search |
| **multi-hop** | 7 | 8 | 15 | Cross-document reasoning |
| **entity-specific** | 7 | 8 | 15 | Rare terms & acronyms |
| **temporal** | 7 | 8 | 15 | Freshness & timestamp tracking |
| **comparison** | 5 | 10 | 15 | Dual-context conflict resolution |
| **TOTAL** | **50** | **70** | **120** | Full Spectrum Evaluation |
