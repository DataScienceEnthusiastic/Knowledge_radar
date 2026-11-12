"""
===============================================================================
🚀 MASTER EXPERIMENT ORCHESTRATOR: Knowledge Radar Evaluation Suite
===============================================================================
Runs all experiments sequentially, records empirical output logs, generates
decision-oriented plots, and verifies end-to-end dataset and metric integrity.

Execution Flow:
  1. EXP-00: Evaluation Integrity & Leakage Audit
  2. EXP-01: Chunking Factorial Optimization (Val Set N=50)
  3. EXP-02: Retrieval Baselines (Dense vs BM25 vs Hybrid) (Val Set N=50)
  4. EXP-03: Reranking Depth & Latency Tradeoff (Val Set N=50)
  5. EXP-04: Web Verification & Conflict Audit (Val Set N=50)
  6. EXP-05: Generation Quality & Groundedness (Val Set N=50)
  7. EXP-06A: Locked Final Test Evaluation (Test Set N=70)
  8. EXP-06B: Post-Hoc Error Waterfall & Failure Taxonomy (Test Set N=70)
  9. Plot Generation (Matplotlib only)
===============================================================================
"""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from evaluation.experiments.exp00_integrity_audit import run_integrity_audit
from evaluation.experiments.exp01_chunking import run_chunking_experiment
from evaluation.experiments.exp01b_chunking_strategies import run_chunking_strategies_experiment
from evaluation.experiments.exp02_retrieval_baselines import run_retrieval_baselines_experiment
from evaluation.experiments.exp03_reranking_depth import run_reranking_depth_experiment
from evaluation.experiments.exp04_web_verification import run_web_verification_experiment
from evaluation.experiments.exp05_generation_quality import run_generation_quality_experiment
from evaluation.experiments.exp06a_final_test_evaluation import run_final_test_evaluation
from evaluation.experiments.exp06b_posthoc_error_waterfall import run_error_waterfall_and_failure_analysis
from evaluation.experiments.plot_generator import generate_all_plots


def main():
    t_start = time.time()
    print("\n" + "=" * 80)
    print("📡 KNOWLEDGE RADAR — COMPLETE EMPIRICAL EVALUATION SUITE")
    print("=" * 80 + "\n")

    # Step 1: Integrity Audit
    audit_res = run_integrity_audit()
    if audit_res["audit_summary"]["status"] != "PASSED":
        print("❌ Integrity Audit Failed! Aborting execution.")
        sys.exit(1)

    # Step 2: Exploratory & Tuning Experiments on Validation Set (N=50)
    run_chunking_experiment()
    run_chunking_strategies_experiment()
    run_retrieval_baselines_experiment()
    run_reranking_depth_experiment()
    run_web_verification_experiment()
    run_generation_quality_experiment()

    # Step 3: Locked Final Test Set Evaluation (N=70)
    run_final_test_evaluation()

    # Step 4: Post-Hoc Error Waterfall & Failure Taxonomy
    run_error_waterfall_and_failure_analysis()

    # Step 5: Generate Visual Figures
    generate_all_plots()

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS.")
    print("All raw CSV logs saved in: evaluation/results/")
    print("All visual figures saved in: evaluation/plots/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
