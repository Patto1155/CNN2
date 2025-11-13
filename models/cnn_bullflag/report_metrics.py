"""CLI to print metrics tables from saved JSON outputs."""

import argparse
import json
from pathlib import Path

DEFAULT_METRICS = Path("models/cnn_bullflag/test_multihead_metrics.json")
DEFAULT_REGIME_METRICS = Path("models/cnn_bullflag/test_multihead_metrics_by_regime.json")


def _print_table(title: str, metrics: dict) -> None:
    print(title)
    header = f"{'Head':<15} {'Th':<4} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Acc':>7}"
    print(header)
    print("-" * len(header))
    for head, thresholds in metrics.items():
        for threshold, values in thresholds.items():
            print(
                f"{head:<15} {threshold:<4} "
                f"{values['precision']:>7.3f} {values['recall']:>7.3f} "
                f"{values['f1']:>7.3f} {values['accuracy']:>7.3f}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Report per-head CNN metrics")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="Path to multihead metrics JSON",
    )
    parser.add_argument(
        "--regime-metrics",
        type=Path,
        default=DEFAULT_REGIME_METRICS,
        help="Path to per-regime metrics JSON",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(args.metrics)

    with args.metrics.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    _print_table("Overall metrics", metrics)

    if args.regime_metrics.exists():
        with args.regime_metrics.open("r", encoding="utf-8") as fh:
            regime_metrics = json.load(fh)
        for regime, regime_values in regime_metrics.items():
            _print_table(f"Regime: {regime}", regime_values)
    else:
        print(f"No regime metrics file at {args.regime_metrics}")


if __name__ == "__main__":
    main()

