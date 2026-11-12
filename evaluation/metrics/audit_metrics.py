"""
===============================================================================
📊 EVALUATION METRICS: Web Verification & Conflict Audit Metrics
===============================================================================
Measures precision and recall of contradiction/conflict detection, temporal
freshness resolution, and extraneous web noise penalty.
===============================================================================
"""

from typing import List, Dict, Any


def compute_conflict_detection_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Computes precision, recall, and F1 for detecting conflicting information
    between internal documents and external web snapshots.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    gt_map = {item["id"]: item.get("has_conflict", False) for item in ground_truth}

    for pred in predictions:
        q_id = pred["id"]
        pred_conflict = pred.get("detected_conflict", False)
        actual_conflict = gt_map.get(q_id, False)

        if pred_conflict and actual_conflict:
            true_positives += 1
        elif pred_conflict and not actual_conflict:
            false_positives += 1
        elif not pred_conflict and actual_conflict:
            false_negatives += 1
        else:
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "conflict_precision": round(precision, 4),
        "conflict_recall": round(recall, 4),
        "conflict_f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives
    }


def compute_temporal_freshness_metrics(
    temporal_predictions: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculates temporal resolution accuracy across time-sensitive queries.
    """
    if not temporal_predictions:
        return {"temporal_accuracy": 0.0, "total_temporal_queries": 0}

    correct_count = sum(1 for p in temporal_predictions if p.get("temporal_resolved_correctly", False))
    total_count = len(temporal_predictions)

    return {
        "temporal_accuracy": round(correct_count / total_count, 4),
        "total_temporal_queries": total_count
    }


def compute_noise_penalty(
    static_predictions: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Measures percentage of extraneous context tokens and latency penalty
    injected when web search is unnecessarily enabled on static queries.
    """
    if not static_predictions:
        return {"avg_injected_noise_tokens": 0.0, "noise_penalty_pct": 0.0}

    noise_tokens_total = sum(p.get("web_context_tokens", 0) for p in static_predictions)
    avg_noise_tokens = noise_tokens_total / len(static_predictions)

    return {
        "avg_injected_noise_tokens": round(avg_noise_tokens, 1),
        "noise_penalty_pct": round((avg_noise_tokens / 500.0) * 100.0, 2)
    }
