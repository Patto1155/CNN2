"""Synthetic smoke test for per-head metrics utilities."""

import json
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._setup_path import add_project_root

add_project_root()

from models.cnn_bullflag.eval_metrics import (
    METRICS_FILENAME,
    REGIME_METRICS_FILENAME,
    save_metrics_to_json,
)
from utils.metrics import compute_multihead_metrics, compute_multihead_metrics_by_regime

OUTPUT_DIR = Path("models/cnn_bullflag")


def build_synthetic(seed: int = 7, n: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = (rng.random((n, 4)) > 0.65).astype(int)
    logits = rng.normal(scale=1.0, size=(n, 4)) + y_true * 1.2
    y_probs = 1 / (1 + np.exp(-logits))
    regimes = rng.choice(["trending_up", "trending_down", "choppy"], size=n)
    return y_true, y_probs, regimes


def print_table(metrics: dict) -> None:
    heads = ["flag", "breakout", "retest", "continuation"]
    header = f"{'Head':<15} {'Th':<4} {'Prec':>6} {'Rec':>6} {'F1':>6}"
    print(header)
    print("-" * len(header))
    for head in heads:
        for threshold, values in metrics[head].items():
            print(
                f"{head:<15} {threshold:<4} "
                f"{values['precision']:>6.3f} {values['recall']:>6.3f} {values['f1']:>6.3f}"
            )


def main() -> None:
    y_true, y_probs, regimes = build_synthetic()
    metrics = compute_multihead_metrics(y_true, y_probs)
    metrics_by_regime = compute_multihead_metrics_by_regime(y_true, y_probs, regimes)

    print("Synthetic multihead metrics (thresholds 0.3/0.5/0.7):")
    print_table(metrics)
    print()
    for regime, regime_metrics in metrics_by_regime.items():
        print(f"Regime: {regime}")
        print_table(regime_metrics)
        print()

    save_metrics_to_json(y_true, y_probs, regimes, output_dir=OUTPUT_DIR)
    print(f"Saved {OUTPUT_DIR / METRICS_FILENAME}")
    print(f"Saved {OUTPUT_DIR / REGIME_METRICS_FILENAME}")

    # Show snippet of JSON for verification
    print("Sample JSON snippet:")
    metrics_path = OUTPUT_DIR / METRICS_FILENAME
    with metrics_path.open("r", encoding="utf-8") as fh:
        head_metrics = json.load(fh)["flag"]
        print(json.dumps({"flag": head_metrics}, indent=2))


if __name__ == "__main__":
    main()
