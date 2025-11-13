"""Plot Grad-CAM activation map alongside/under a price series.

Reads a JSONL produced by scripts/generate_cnn_output_demo.py (or your pipeline),
selects a window by id or index, and renders:
  - Top: price line (from CSV optional; otherwise a synthetic walk)
  - Bottom: activation intensities as a heat bar and line

Usage examples:
  python scripts/plot_gradcam_overlay.py \
      --jsonl models/cnn_bullflag/example_cnn_outputs.jsonl \
      --index 0 \
      --out models/cnn_bullflag/plots/window_0000.png

  python scripts/plot_gradcam_overlay.py \
      --jsonl your_outputs.jsonl --window-id window_0123 \
      --price-csv path/to/prices.csv --price-col close
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    # Use non-interactive backend by default (for headless execution)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Matplotlib is required for plotting. Install via 'pip install matplotlib'"
    ) from e


def load_jsonl(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {jsonl_path}")
    return rows


def pick_row(rows: list[dict], index: Optional[int], window_id: Optional[str]) -> dict:
    if window_id is not None:
        for r in rows:
            if r.get("window_id") == window_id:
                return r
        raise KeyError(f"window_id {window_id} not found in JSONL")
    if index is None:
        index = 0
    if index < 0 or index >= len(rows):
        raise IndexError(f"index {index} out of range (0..{len(rows)-1})")
    return rows[index]


def load_price_series(
    csv_path: Optional[Path], col: str, length: int
) -> tuple[np.ndarray, np.ndarray]:
    if csv_path is None:
        # Synthetic price: random walk starting at 100
        rng = np.random.default_rng(0)
        steps = rng.normal(scale=0.3, size=length)
        price = 100 + np.cumsum(steps)
        x = np.arange(length)
        return x, price
    import pandas as pd  # optional dependency; common for CSV

    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in {csv_path}. Available: {list(df.columns)}")
    series = df[col].to_numpy(dtype=float)
    # Align to desired length
    if series.size >= length:
        series = series[-length:]
    else:
        # pad at start
        pad = np.full(length - series.size, series[0])
        series = np.concatenate([pad, series])
    x = np.arange(length)
    return x, series


def plot_overlay(row: dict, price_x: np.ndarray, price_y: np.ndarray, out: Path) -> None:
    intensities = np.array(row["activation_map"]["intensities"], dtype=float)
    idx = np.array(row["activation_map"]["indices"], dtype=int)
    assert intensities.shape[0] == idx.shape[0]

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.5, 1.2], hspace=0.25)

    ax_price = fig.add_subplot(gs[0, 0])
    ax_price.plot(price_x, price_y, color="#1f77b4", label="price")
    ax_price.set_title(
        f"{row['symbol']} {row['tf']} — {row['window_id']} (seq={row['scores']['sequence_score']:.3f})"
    )
    ax_price.set_xlim(0, intensities.size - 1)
    ax_price.grid(True, alpha=0.2)
    ax_price.legend(loc="upper left")

    ax_cam = fig.add_subplot(gs[1, 0])
    # Heat bar using imshow
    ax_cam.imshow(
        intensities[np.newaxis, :],
        aspect="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        extent=[0, intensities.size - 1, 0, 1],
    )
    ax_cam.plot(idx, intensities, color="white", linewidth=1.0, alpha=0.8)
    ax_cam.set_yticks([])
    ax_cam.set_xlim(0, intensities.size - 1)
    ax_cam.set_xlabel("bar index")
    ax_cam.set_title("Grad‑CAM activation (0..1)")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Grad-CAM overlay for one window")
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to JSONL outputs")
    parser.add_argument("--index", type=int, help="Row index (default 0)")
    parser.add_argument("--window-id", type=str, help="Select by window_id instead of index")
    parser.add_argument("--price-csv", type=Path, help="CSV with price column")
    parser.add_argument("--price-col", type=str, default="close", help="CSV column for price")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    rows = load_jsonl(args.jsonl)
    row = pick_row(rows, args.index, args.window_id)
    if row.get("activation_map") is None:
        raise ValueError("Selected row has no activation_map. Re-run inference with Grad-CAM enabled.")

    seq_len = len(row["activation_map"]["intensities"])
    x, price = load_price_series(args.price_csv, args.price_col, seq_len)
    plot_overlay(row, x, price, args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

