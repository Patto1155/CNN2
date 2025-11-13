"""Helpers for computing and persisting per-head evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from utils.metrics import (
    compute_multihead_metrics,
    compute_multihead_metrics_by_regime,
)

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.3, 0.5, 0.7)
DEFAULT_OUTPUT_DIR = Path("models/cnn_bullflag")
METRICS_FILENAME = "test_multihead_metrics.json"
REGIME_METRICS_FILENAME = "test_multihead_metrics_by_regime.json"


def save_metrics_to_json(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    regimes: Optional[np.ndarray] = None,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Compute metrics and persist them to JSON files in output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_multihead_metrics(y_true, y_probs, thresholds)
    with (output_dir / METRICS_FILENAME).open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    if regimes is not None:
        metrics_by_regime = compute_multihead_metrics_by_regime(
            y_true, y_probs, regimes, thresholds
        )
        with (output_dir / REGIME_METRICS_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(metrics_by_regime, fh, indent=2)


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "arr_0" in data:
            return data["arr_0"]
        raise ValueError(f"NPZ file {path} must contain 'arr_0'")
    raise ValueError(f"Unsupported file extension for {path}")


def _generate_synthetic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = (rng.random((n, 4)) > 0.7).astype(int)
    logits = rng.normal(size=(n, 4)) + y_true * 1.5
    y_probs = 1 / (1 + np.exp(-logits))
    regimes = rng.choice(["trending_up", "trending_down", "choppy"], size=n)
    return y_true, y_probs, regimes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute multihead metrics and save to JSON")
    parser.add_argument("--ytrue", type=Path, help="Path to y_true .npy/.npz", nargs="?")
    parser.add_argument("--yprobs", type=Path, help="Path to y_prob .npy/.npz", nargs="?")
    parser.add_argument("--regimes", type=Path, help="Path to regimes .npy/.npz", nargs="?")
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="Generate synthetic dataset of this size instead of loading files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store metrics JSON",
    )
    args = parser.parse_args()

    if args.synthetic > 0:
        y_true, y_probs, regimes = _generate_synthetic(args.synthetic)
    else:
        if args.ytrue is None or args.yprobs is None:
            parser.error("Provide --synthetic or both --ytrue and --yprobs paths")
        y_true = _load_array(args.ytrue)
        y_probs = _load_array(args.yprobs)
        regimes = _load_array(args.regimes) if args.regimes else None

    save_metrics_to_json(
        y_true=y_true,
        y_probs=y_probs,
        regimes=regimes,
        output_dir=args.output_dir,
    )
    print(f"Saved metrics to {args.output_dir / METRICS_FILENAME}")
    if regimes is not None:
        print(f"Saved regime metrics to {args.output_dir / REGIME_METRICS_FILENAME}")


if __name__ == "__main__":
    main()

