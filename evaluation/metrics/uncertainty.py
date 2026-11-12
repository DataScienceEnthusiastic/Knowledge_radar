"""
===============================================================================
📊 EVALUATION METRIC: Bootstrap Uncertainty Estimation
===============================================================================
Computes empirical 95% bootstrap confidence intervals for major retrieval
and generation metrics.
===============================================================================
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def bootstrap_confidence_interval(
    values: List[float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes empirical bootstrap confidence intervals for a metric series.
    
    Args:
        values: List of metric values (one per evaluated query).
        n_bootstraps: Number of bootstrap resamples (default: 1000).
        ci: Confidence interval span (default: 0.95).
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        Dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'sample_size'.
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "sample_size": 0
        }

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 1:
        val = float(arr[0])
        return {
            "mean": val,
            "std": 0.0,
            "ci_lower": val,
            "ci_upper": val,
            "sample_size": 1
        }

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(resample)

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    ci_lower = float(np.percentile(bootstrap_means, lower_pct))
    ci_upper = float(np.percentile(bootstrap_means, upper_pct))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "sample_size": n
    }
