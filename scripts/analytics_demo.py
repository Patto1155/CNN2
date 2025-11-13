"""Simple analytics demo ranking windows and summarizing activation maps."""

import json
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._setup_path import add_project_root

add_project_root()

from models.cnn_bullflag.schema import build_cnn_output

OUTPUTS_PATH = Path("models/cnn_bullflag/example_cnn_outputs.jsonl")


def _load_outputs() -> list[dict]:
    if not OUTPUTS_PATH.exists():
        raise FileNotFoundError(
            f"{OUTPUTS_PATH} not found. Run scripts/generate_cnn_output_demo.py first."
        )
    rows = []
    with OUTPUTS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict], top_n: int = 3) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["scores"]["sequence_score"], reverse=True)
    print(f"Top {top_n} windows by sequence_score:")
    for row in rows_sorted[:top_n]:
        scores = row["scores"]
        print(
            f"{row['window_id']}: seq={scores['sequence_score']:.3f} "
            f"flag={scores['flag_prob']:.3f} breakout={scores['breakout_prob']:.3f}"
        )

    print()
    print("Activation focus (top 5 indices by intensity):")
    for row in rows_sorted[:top_n]:
        activation = row.get("activation_map")
        if not activation:
            continue
        intensities = np.array(activation["intensities"])
        top_indices = intensities.argsort()[-5:][::-1]
        print(f"{row['window_id']}: indices {top_indices.tolist()} with intensities {intensities[top_indices].round(3).tolist()}")


def main() -> None:
    rows = _load_outputs()
    summarize(rows)


if __name__ == "__main__":
    main()
