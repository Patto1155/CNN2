#!/usr/bin/env python3
"""
Visualize randomly sampled labeled windows with event markers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._setup_path import add_project_root

add_project_root()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts import label_windows  # noqa: E402


def _load_labeled_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "y" not in data.files:
        raise ValueError(f"{path} is missing labels ('y'). Run scripts/label_windows.py first.")
    return data["X"], data["y"]


def _plot_window(
    window: np.ndarray,
    diag: label_windows.LabelDiagnostics,
    label_vec: np.ndarray,
    out_path: Path,
) -> None:
    close = window[label_windows.CHANNELS["close"]]
    high = window[label_windows.CHANNELS["high"]]
    low = window[label_windows.CHANNELS["low"]]
    ema20 = window[label_windows.CHANNELS["ema20"]]
    ema50 = window[label_windows.CHANNELS["ema50"]]
    x = np.arange(close.size)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, close, label="close", linewidth=1.2)
    ax.plot(x, ema20, label="ema20", linewidth=0.9)
    ax.plot(x, ema50, label="ema50", linewidth=0.9)

    marker_specs = [
        ("flag", [diag.t_flag_end] if diag.t_flag_end is not None else [], close, "tab:orange", "o"),
        ("breakout", [diag.t_breakout] if diag.t_breakout is not None else [], close, "tab:green", "^"),
        ("retest", [diag.t_retest] if diag.t_retest is not None else [], close, "tab:red", "v"),
        ("continuation", [diag.t_continuation] if diag.t_continuation is not None else [], high, "tab:blue", "s"),
    ]
    for name, positions, series, color, marker in marker_specs:
        if not positions:
            continue
        x_vals = [pos for pos in positions if 0 <= pos < series.size]
        if not x_vals:
            continue
        y_vals = [series[pos] for pos in x_vals]
        ax.scatter(x_vals, y_vals, color=color, marker=marker, label=name, zorder=5)

    label_text = ", ".join(
        f"{name}={label_vec[i]}" for i, name in enumerate(label_windows.LABEL_NAMES)
    )
    ax.set_title(label_text)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Price")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview labeled windows with markers.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset_labeled.npz",
        help="NPZ file containing X and y arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/cnn_bullflag/plots/label_debug",
        help="Directory to store preview images.",
    )
    parser.add_argument("--num-samples", type=int, default=6, help="Number of windows to plot.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    X, y = _load_labeled_dataset(dataset_path)
    params = label_windows.load_labeler_params()

    positive_indices = np.where(y.sum(axis=1) > 0)[0]
    if positive_indices.size == 0:
        raise ValueError("No positive labels found; run the labeler first.")

    sample_count = min(args.num_samples, positive_indices.size)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(positive_indices, size=sample_count, replace=False)

    out_dir = Path(args.out_dir)
    saved_paths: list[Path] = []
    for idx in sample_indices:
        window = X[idx]
        label_vec, diag = label_windows.label_single_window(window, params=params)
        out_path = out_dir / f"window_{idx:05d}.png"
        _plot_window(window, diag, label_vec, out_path)
        saved_paths.append(out_path)

    print(f"Saved {len(saved_paths)} label previews to {out_dir}")


if __name__ == "__main__":
    main()
