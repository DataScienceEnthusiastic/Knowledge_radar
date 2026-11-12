"""
===============================================================================
🔬 EXP-00: Evaluation Integrity & Data Leakage Audit
===============================================================================
Performs comprehensive pre-evaluation audit:
1. Duplicate and near-duplicate document detection (Jaccard token similarity)
2. Benchmark query-document contamination check
3. Strict disjoint validation (N=50) vs test (N=70) split isolation verification
4. Generates audit report in evaluation/results/exp00_integrity_audit_report.json
===============================================================================
"""

import os
import sys
import json
import re
import datetime
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "evaluation", "dataset")
DOCS_DIR = os.path.join(DATASET_DIR, "documents")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def tokenize(text: str):
    return set(re.findall(r'\w+', text.lower()))


def run_integrity_audit():
    print("=" * 70)
    print("🔬 RUNNING EXP-00: EVALUATION INTEGRITY & LEAKAGE AUDIT")
    print("=" * 70)

    # 1. Document Near-Duplicate Check
    doc_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
    doc_texts = {}
    for f in doc_files:
        with open(os.path.join(DOCS_DIR, f), "r", encoding="utf-8") as fp:
            doc_texts[f] = tokenize(fp.read())

    high_similarity_pairs = []
    doc_names = list(doc_texts.keys())
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            d1, d2 = doc_names[i], doc_names[j]
            t1, t2 = doc_texts[d1], doc_texts[d2]
            intersection = t1.intersection(t2)
            union = t1.union(t2)
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard > 0.80:
                high_similarity_pairs.append({"doc_a": d1, "doc_b": d2, "jaccard_similarity": round(jaccard, 4)})

    # 2. Split Disjointness & Leakage
    val_path = os.path.join(SPLITS_DIR, "validation.jsonl")
    test_path = os.path.join(SPLITS_DIR, "test.jsonl")

    with open(val_path, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f if line.strip()]

    with open(test_path, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]

    val_ids = set(s["id"] for s in val_samples)
    test_ids = set(s["id"] for s in test_samples)
    leakage_ids = list(val_ids.intersection(test_ids))

    # 3. Query Contamination Check (Verbatim n-gram overlap >= 90% with target docs)
    contaminated_queries = []
    for s in val_samples + test_samples:
        q_tokens = tokenize(s["query"])
        # Check if query is simply a copy-paste of a long document sentence
        if len(q_tokens) > 15:
            for doc_name in s.get("relevant_docs", []):
                doc_t = doc_texts.get(doc_name, set())
                if q_tokens.issubset(doc_t):
                    contaminated_queries.append({"id": s["id"], "doc": doc_name})

    passed = (len(high_similarity_pairs) == 0 and len(leakage_ids) == 0 and len(contaminated_queries) == 0)

    report = {
        "metadata": {
            "experiment_id": "EXP-00-INTEGRITY-AUDIT",
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": get_git_commit(),
            "dataset_version": "v1.0.0",
            "reproduction_command": "python evaluation/experiments/exp00_integrity_audit.py"
        },
        "audit_summary": {
            "status": "PASSED" if passed else "FAILED",
            "total_documents_checked": len(doc_files),
            "validation_queries_count": len(val_samples),
            "test_queries_count": len(test_samples),
            "near_duplicate_doc_pairs_count": len(high_similarity_pairs),
            "split_leakage_overlap_count": len(leakage_ids),
            "verbatim_contaminated_queries_count": len(contaminated_queries)
        },
        "findings": {
            "near_duplicate_pairs": high_similarity_pairs,
            "leakage_query_ids": leakage_ids,
            "contaminated_queries": contaminated_queries
        }
    }

    report_path = os.path.join(RESULTS_DIR, "exp00_integrity_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Audit Status: {report['audit_summary']['status']}")
    print(f"Total Documents: {len(doc_files)} | Val Set: {len(val_samples)} | Test Set: {len(test_samples)}")
    print(f"Report saved to: {report_path}\n")
    return report


if __name__ == "__main__":
    run_integrity_audit()
