1. Per-head metrics: “is it good at flags or just breakouts?”

You have:

Predictions: y_pred_probs → shape [N, 4] for [flag, breakout, retest, continuation]

Ground truth: y_true → same shape

You want metrics per head:

TP / FP / TN / FN

Precision / Recall / F1 / Accuracy

Optionally metrics at multiple thresholds (0.3, 0.5, 0.7)

Optionally per regime (trending / choppy / etc.)

1.1 Core metric function per head

Spec for metrics.py:

from typing import Dict, List
import numpy as np

def compute_binary_metrics(
    y_true: np.ndarray,   # shape [N], 0/1
    y_prob: np.ndarray,   # shape [N], floats in [0,1]
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics for one head.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

1.2 Metrics for all heads, all thresholds
HEAD_NAMES = ["flag", "breakout", "retest", "continuation"]

def compute_multihead_metrics(
    y_true: np.ndarray,       # [N, 4]
    y_probs: np.ndarray,      # [N, 4]
    thresholds: List[float] = (0.3, 0.5, 0.7),
) -> Dict[str, Dict]:
    """
    Returns nested dict:
    {
      "flag": {
        "0.3": {...},
        "0.5": {...},
        "0.7": {...}
      },
      "breakout": {...},
      ...
    }
    """
    metrics = {}
    for head_idx, head_name in enumerate(HEAD_NAMES):
        metrics[head_name] = {}
        y_true_head = y_true[:, head_idx]
        y_prob_head = y_probs[:, head_idx]

        for th in thresholds:
            m = compute_binary_metrics(y_true_head, y_prob_head, threshold=th)
            metrics[head_name][str(th)] = m

    return metrics


You then save this to JSON:

save_json(metrics, "models/cnn_bullflag/test_multihead_metrics.json")

1.3 Per-regime metrics (optional, but useful)

Assume you also have regimes array [N] with strings like "trending_up", "choppy", etc.

def compute_multihead_metrics_by_regime(
    y_true: np.ndarray, y_probs: np.ndarray, regimes: np.ndarray,
    thresholds=(0.5,)
):
    results = {}
    unique_regimes = sorted(set(regimes))

    for regime in unique_regimes:
        mask = (regimes == regime)
        if mask.sum() == 0:
            continue
        metrics_regime = compute_multihead_metrics(
            y_true[mask], y_probs[mask], thresholds
        )
        results[regime] = metrics_regime

    return results


This gives you exactly: “CNN is good at flags in trends, trash at retests in chop”.