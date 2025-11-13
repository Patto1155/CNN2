"""Generate a multi-window Grad-CAM gallery.

Selects top-N windows by a score (sequence_score or a specific head),
and saves per-window PNGs plus a simple HTML index.

Examples:
  python scripts/plot_gallery.py \
      --jsonl models/cnn_bullflag/example_cnn_outputs.jsonl \
      --out-dir models/cnn_bullflag/plots/gallery \
      --top-n 12 --sort-by sequence_score

  python scripts/plot_gallery.py --jsonl your.jsonl --sort-by flag_prob --top-n 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit("Matplotlib is required. 'pip install matplotlib'") from e

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def load_rows(jsonl: Path) -> List[dict]:
    rows: List[dict] = []
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {jsonl}")
    return rows


def pick_top(rows: List[dict], sort_by: str, top_n: int) -> List[dict]:
    def key_fn(r: dict) -> float:
        if sort_by == "sequence_score":
            return float(r["scores"]["sequence_score"])  # type: ignore[index]
        if sort_by in ("flag_prob", "breakout_prob", "retest_prob", "continuation_prob"):
            return float(r["scores"][sort_by])  # type: ignore[index]
        raise KeyError(f"Unknown sort_by '{sort_by}'")

    return sorted(rows, key=key_fn, reverse=True)[:top_n]


def plot_one(row: dict, out: Path, price: np.ndarray | None = None) -> None:
    intensities = np.array(row["activation_map"]["intensities"], dtype=float)
    x = np.arange(intensities.size)
    if price is None:
        rng = np.random.default_rng(0)
        price = 100 + np.cumsum(rng.normal(scale=0.3, size=intensities.size))

    fig = plt.figure(figsize=(6, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, price, color="#1f77b4")
    ax1.set_xlim(0, intensities.size - 1)
    ax1.grid(True, alpha=0.2)
    title = f"{row['window_id']} seq={row['scores']['sequence_score']:.3f}"
    ax1.set_title(title)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(intensities[np.newaxis, :], aspect="auto", cmap="inferno", vmin=0, vmax=1,
               extent=[0, intensities.size - 1, 0, 1])
    ax2.plot(x, intensities, color="white", linewidth=1.0)
    ax2.set_yticks([])
    ax2.set_xlim(0, intensities.size - 1)
    ax2.set_xlabel("bar index")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


def save_index_html(rows: List[dict], out_dir: Path, sort_by: str) -> None:
    html = [
        "<html><head><meta charset='utf-8'><title>Grad-CAM Gallery</title>",
        "<style> body{font-family:Arial, sans-serif;} .card{display:inline-block;margin:8px;} img{border:1px solid #ddd;}</style>",
        "</head><body>",
        f"<h2>Grad-CAM Gallery (sorted by {sort_by})</h2>",
    ]
    for r in rows:
        img = f"{r['window_id']}.png"
        caption = (
            f"{r['window_id']} | seq={r['scores']['sequence_score']:.3f} "
            f"flag={r['scores']['flag_prob']:.3f} brk={r['scores']['breakout_prob']:.3f} "
            f"ret={r['scores']['retest_prob']:.3f} cont={r['scores']['continuation_prob']:.3f}"
        )
        html.append(f"<div class='card'><img src='{img}' width='340'><br><small>{caption}</small></div>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a multi-window Grad-CAM gallery")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument(
        "--sort-by",
        type=str,
        default="sequence_score",
        choices=["sequence_score", "flag_prob", "breakout_prob", "retest_prob", "continuation_prob"],
    )
    parser.add_argument("--price-csv", type=Path, help="Optional price CSV")
    parser.add_argument("--price-col", type=str, default="close")
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    # Filter to rows that actually have activation maps
    rows = [r for r in rows if r.get("activation_map")]
    if not rows:
        raise ValueError("No rows with activation_map found; re-run inference with Grad-CAM enabled.")
    picked = pick_top(rows, args.sort_by, args.top_n)

    # Load optional price series if provided. If the CSV doesn't match length, fallback to synthetic per-window.
    price = None
    if args.price_csv is not None:
        try:
            import pandas as pd

            df = pd.read_csv(args.price_csv)
            if args.price_col not in df.columns:
                raise KeyError
            vec = df[args.price_col].to_numpy(dtype=float)
            # we'll slice per-window later using last L points
            price = vec
        except Exception:
            price = None

    it = picked
    if tqdm is not None:
        it = tqdm(picked, desc="Rendering", unit="img")  # type: ignore
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for r in it:  # type: ignore
        L = len(r["activation_map"]["intensities"])
        if price is not None and price.size >= L:
            p = price[-L:]
        else:
            p = None
        out = args.out_dir / f"{r['window_id']}.png"
        plot_one(r, out, p)

    save_index_html(picked, args.out_dir, args.sort_by)
    print(f"Saved {len(picked)} images and index.html in {args.out_dir}")


if __name__ == "__main__":
    main()

