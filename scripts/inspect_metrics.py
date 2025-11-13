"""Inspect saved metrics JSON and highlight potential issues."""

import json
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._setup_path import add_project_root

add_project_root()

DEFAULT_PATH = Path("models/cnn_bullflag/test_multihead_metrics.json")


def main() -> None:
    if not DEFAULT_PATH.exists():
        raise FileNotFoundError(
            f"{DEFAULT_PATH} not found. Run scripts/test_metrics_synthetic.py first."
        )
    with DEFAULT_PATH.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    for head, thresholds in metrics.items():
        avg_f1 = sum(m["f1"] for m in thresholds.values()) / len(thresholds)
        best_th, best_metrics = max(
            thresholds.items(), key=lambda kv: kv[1]["f1"]
        )
        print(
            f"{head:<15} avg F1={avg_f1:.3f} best@{best_th} "
            f"Prec={best_metrics['precision']:.3f} Rec={best_metrics['recall']:.3f}"
        )
        if best_metrics["recall"] < 0.5:
            print(f"  - Consider lowering threshold for {head} to improve recall")
        if best_metrics["precision"] < 0.5:
            print(f"  - Consider tightening labels or threshold for {head}")


if __name__ == "__main__":
    main()
