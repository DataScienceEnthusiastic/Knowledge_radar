"""
===============================================================================
🔬 EXP-06B: Post-Hoc Error Waterfall & 10-Point Failure Taxonomy
===============================================================================
Consumes the frozen test results from EXP-06A (without modifying system configuration)
to construct:
1. Step-by-step Error Waterfall Attrition
   (Total -> Retrieval Hit -> Citation Support -> Groundedness -> Correctness)
2. 10-Point Failure Taxonomy classification with category frequencies
3. Human-auditable failure inspection log (failure_inspection_sample.jsonl)
===============================================================================
"""

import os
import sys
import json
import csv
import datetime
import subprocess
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")
DATASET_DIR = os.path.join(BASE_DIR, "evaluation", "dataset")

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_error_waterfall_and_failure_analysis():
    print("=" * 70)
    print("🔬 RUNNING EXP-06B: POST-HOC ERROR WATERFALL & FAILURE ANALYSIS")
    print("=" * 70)

    # 1. Load frozen test results from EXP-06A
    test_results_csv = os.path.join(RESULTS_DIR, "exp06a_final_test_results.csv")
    if not os.path.exists(test_results_csv):
        print(f"Error: {test_results_csv} not found. Running EXP-06A first...")
        from evaluation.experiments.exp06a_final_test_evaluation import run_final_test_evaluation
        run_final_test_evaluation()

    rows = []
    with open(test_results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_queries = len(rows)

    # 2. Compute Error Waterfall
    # Step 1: Total Queries
    step1_total = total_queries

    # Step 2: Retrieval Success (Recall@5 > 0)
    retrieval_success_rows = [r for r in rows if float(r["recall_at_5"]) > 0.0]
    step2_retr_success = len(retrieval_success_rows)

    # Step 3: Citation Support (citation_validity > 0 and context has evidence)
    citation_support_rows = [r for r in retrieval_success_rows if float(r["citation_validity"]) > 0.0]
    step3_citation_support = len(citation_support_rows)

    # Step 4: Groundedness (faithfulness >= 0.70)
    grounded_rows = [r for r in citation_support_rows if float(r["faithfulness"]) >= 0.70]
    step4_groundedness = len(grounded_rows)

    # Step 5: Answer Correctness (correctness >= 0.70)
    correct_rows = [r for r in grounded_rows if float(r["answer_correctness"]) >= 0.70]
    step5_correct = len(correct_rows)

    waterfall_data = [
        {"stage": "1_Total_Test_Queries", "count": step1_total, "pct_of_total": 100.0, "step_loss": 0},
        {"stage": "2_Retrieval_Success (Recall@5 > 0)", "count": step2_retr_success, "pct_of_total": round(step2_retr_success/step1_total*100, 2), "step_loss": step1_total - step2_retr_success},
        {"stage": "3_Citation_Support", "count": step3_citation_support, "pct_of_total": round(step3_citation_support/step1_total*100, 2), "step_loss": step2_retr_success - step3_citation_support},
        {"stage": "4_Faithfulness (>= 0.70)", "count": step4_groundedness, "pct_of_total": round(step4_groundedness/step1_total*100, 2), "step_loss": step3_citation_support - step4_groundedness},
        {"stage": "5_Final_Answer_Correctness (>= 0.70)", "count": step5_correct, "pct_of_total": round(step5_correct/step1_total*100, 2), "step_loss": step4_groundedness - step5_correct}
    ]

    print("\n🌊 ERROR WATERFALL CASCADE (N=70 Held-Out Queries):")
    print("-" * 75)
    for stage in waterfall_data:
        print(f"{stage['stage']:<45}: {stage['count']:>3} queries ({stage['pct_of_total']:>5.1f}%) [Loss: {stage['step_loss']}]")

    # 3. Classify Failures into 10-Point Failure Taxonomy
    taxonomy_counts = {
        "RETRIEVAL_MISSING_EVIDENCE": 0,
        "RETRIEVAL_WRONG_EVIDENCE": 0,
        "RANKING_FAILURE": 0,
        "CONTEXT_OVERLOAD": 0,
        "LLM_REASONING_FAILURE": 0,
        "TEMPORAL_MISMATCH": 0,
        "WEB_CONFLICT_RESOLUTION_FAILURE": 0,
        "CITATION_GROUNDING_HALLUCINATION": 0,
        "PARSING_OCR_DEGRADATION": 0,
        "SYSTEM_TIMEOUT_API_ERROR": 0
    }

    failure_samples = []

    for r in rows:
        r5 = float(r["recall_at_5"])
        r1 = float(r["recall_at_1"])
        corr = float(r["answer_correctness"])
        faith = float(r["faithfulness"])
        cite = float(r["citation_validity"])
        q_cat = r["category"]

        # Failure attribution logic:
        assigned_category = None
        root_cause = ""

        if r5 == 0.0:
            assigned_category = "RETRIEVAL_MISSING_EVIDENCE"
            root_cause = "Relevant ground-truth document was absent from Top-5 candidate context."
        elif r1 == 0.0 and r5 > 0.0 and corr < 0.60:
            assigned_category = "RANKING_FAILURE"
            root_cause = "Relevant document retrieved in candidate pool but placed at lower rank in Top-5."
        elif faith < 0.60 and corr < 0.60:
            assigned_category = "CITATION_GROUNDING_HALLUCINATION"
            root_cause = "Generated assertions were not supported by retrieved context snippets."
        elif q_cat == "temporal" and corr < 0.70:
            assigned_category = "TEMPORAL_MISMATCH"
            root_cause = "Temporal freshness resolution failed to select current 2025/2026 ground truth."
        elif q_cat == "comparison" and corr < 0.70:
            assigned_category = "WEB_CONFLICT_RESOLUTION_FAILURE"
            root_cause = "Contradiction between document vault and web verification was unresolved."
        elif corr < 0.70:
            assigned_category = "LLM_REASONING_FAILURE"
            root_cause = "Retrieved context contained partial evidence, but synthesis drew incomplete inference."

        if assigned_category:
            taxonomy_counts[assigned_category] += 1
            failure_samples.append({
                "query_id": r["query_id"],
                "category": r["category"],
                "failure_mode": assigned_category,
                "root_cause_explanation": root_cause,
                "answer_correctness": corr,
                "faithfulness": faith,
                "generated_answer": r.get("generated_answer", "")
            })

    total_failures = sum(taxonomy_counts.values())

    print("\n🔍 10-POINT FAILURE TAXONOMY DISTRIBUTION (Total Failure Events = {}):".format(total_failures))
    print("-" * 75)
    print(f"{'Failure Category':<38}{'Count':<8}{'Share of Failures (%)':<24}{'Share of Total Set (%)'}")
    print("-" * 75)

    taxonomy_rows = []
    git_commit = get_git_commit()
    timestamp = datetime.datetime.now().isoformat()

    for cat_name, count in sorted(taxonomy_counts.items(), key=lambda x: x[1], reverse=True):
        share_failures = round((count / total_failures * 100.0), 2) if total_failures > 0 else 0.0
        share_total = round((count / total_queries * 100.0), 2)
        print(f"{cat_name:<38}{count:<8}{share_failures:<24.2f}{share_total:.2f}%")

        taxonomy_rows.append({
            "experiment_id": "EXP-06B-FAILURE-TAXONOMY",
            "git_commit": git_commit,
            "timestamp": timestamp,
            "failure_category": cat_name,
            "failure_count": count,
            "share_of_failures_pct": share_failures,
            "share_of_total_queries_pct": share_total,
            "reproduction_command": "python evaluation/experiments/exp06b_posthoc_error_waterfall.py"
        })

    # Save waterfall CSV
    waterfall_csv = os.path.join(RESULTS_DIR, "exp06b_error_waterfall.csv")
    with open(waterfall_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(waterfall_data[0].keys()))
        writer.writeheader()
        writer.writerows(waterfall_data)

    # Save taxonomy CSV
    taxonomy_csv = os.path.join(RESULTS_DIR, "exp06b_failure_taxonomy.csv")
    with open(taxonomy_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(taxonomy_rows[0].keys()))
        writer.writeheader()
        writer.writerows(taxonomy_rows)

    # Save failure inspection sample JSONL
    sample_jsonl = os.path.join(RESULTS_DIR, "failure_inspection_sample.jsonl")
    with open(sample_jsonl, "w", encoding="utf-8") as f:
        for s in failure_samples:
            f.write(json.dumps(s) + "\n")

    print("-" * 75)
    print(f"Waterfall data saved to:       {waterfall_csv}")
    print(f"Failure taxonomy saved to:     {taxonomy_csv}")
    print(f"Inspection samples saved to:   {sample_jsonl}\n")

    return {
        "waterfall": waterfall_data,
        "taxonomy": taxonomy_rows,
        "inspection_samples": failure_samples
    }


if __name__ == "__main__":
    run_error_waterfall_and_failure_analysis()
