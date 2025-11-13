from typing import Any, Dict, Iterable, List, Mapping

import numpy as np


HEAD_NAMES: List[str] = ["flag", "breakout", "retest", "continuation"]


def _to_bool_array(x: np.ndarray) -> np.ndarray:
    """Ensure boolean/int array of shape [N]."""
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array for y_true; got shape {x.shape}")
    # Cast to int -> bool
    return (x.astype(int) != 0)


def _finite_mask(*arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Return a boolean mask that is True where all arrays are finite."""
    masks = []
    for arr in arrays:
        masks.append(np.isfinite(arr))
    mask = masks[0]
    for m in masks[1:]:
        mask = mask & m
    return mask


def compute_binary_metrics(
    y_true: np.ndarray,  # shape [N], 0/1
    y_prob: np.ndarray,  # shape [N], floats in [0,1]
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute standard binary classification metrics for one head.
    Handles NaN/inf in y_prob via masking (excluded from counts).
    Returns only Python-native types friendly to JSON serialization.
    """
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError(
            f"y_true and y_prob must be 1D; got {y_true.shape} and {y_prob.shape}"
        )

    mask = _finite_mask(y_prob)
    if mask.sum() == 0:
        # no valid data; return zeros
        return {
            "threshold": float(threshold),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    y_true_b = _to_bool_array(y_true[mask])
    y_prob_v = y_prob[mask].astype(float)

    y_pred = (y_prob_v >= float(threshold))

    tp = int(((y_pred == True) & (y_true_b == True)).sum())
    fp = int(((y_pred == True) & (y_true_b == False)).sum())
    tn = int(((y_pred == False) & (y_true_b == False)).sum())
    fn = int(((y_pred == False) & (y_true_b == True)).sum())

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def compute_multihead_metrics(
    y_true: np.ndarray,  # [N, 4]
    y_probs: np.ndarray,  # [N, 4]
    thresholds: Iterable[float] = (0.3, 0.5, 0.7),
) -> Dict[str, Dict[str, Dict[str, Any]]]:
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
    if y_true.ndim != 2 or y_true.shape[1] != 4:
        raise ValueError(f"y_true must be [N, 4]; got {y_true.shape}")
    if y_probs.ndim != 2 or y_probs.shape[1] != 4:
        raise ValueError(f"y_probs must be [N, 4]; got {y_probs.shape}")
    if y_true.shape[0] != y_probs.shape[0]:
        raise ValueError("y_true and y_probs must have same N")

    metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for head_idx, head_name in enumerate(HEAD_NAMES):
        metrics[head_name] = {}
        y_true_head = y_true[:, head_idx].astype(int)
        y_prob_head = y_probs[:, head_idx].astype(float)
        for th in thresholds:
            m = compute_binary_metrics(y_true_head, y_prob_head, threshold=float(th))
            metrics[head_name][str(float(th))] = m
    return metrics


def compute_multihead_metrics_by_regime(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    regimes: np.ndarray,
    thresholds: Iterable[float] = (0.5,),
) -> Dict[str, Mapping[str, Dict[str, Dict[str, Any]]]]:
    """
    Compute metrics per regime label. Regimes can be strings or numbers.
    Returns dict: { regime: compute_multihead_metrics(...), ... }
    """
    if regimes.ndim != 1 or regimes.shape[0] != y_true.shape[0]:
        raise ValueError("regimes must be [N] and aligned with y_true/y_probs")

    results: Dict[str, Mapping[str, Dict[str, Dict[str, Any]]]] = {}
    # Convert to Python scalars for robust set/sort
    regimes_py = [str(r) for r in regimes.tolist()]
    unique_regimes = sorted(set(regimes_py))

    regimes_arr = np.array(regimes_py)
    for regime in unique_regimes:
        mask = regimes_arr == regime
        if mask.sum() == 0:
            continue
        metrics_regime = compute_multihead_metrics(
            y_true[mask], y_probs[mask], thresholds
        )
        results[regime] = metrics_regime
    return results

