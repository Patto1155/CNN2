"""
Config-driven inference scaffold for BullFlag CNN.

Loads a trained checkpoint and runs inference on either:
  - a directory of windows (expects NPZ/NPY with X: [N,C,L]), or
  - sliding windows generated from a CSV with numeric columns.

Writes JSONL to [inference].output_jsonl and optionally includes Grad-CAM
activation maps when [inference].enable_gradcam is true.

Run:
  python scripts/infer_real.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from scripts._setup_path import add_project_root


add_project_root()

from scripts.config import load_config  # noqa: E402
from models.cnn_bullflag.model import build_model_from_config  # noqa: E402
from models.cnn_bullflag.infer import (  # noqa: E402
    BullFlagCNNInfer,
    WindowMeta,
)

try:  # Optional dependences
    import pandas as pd
except Exception:
    pd = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore


def _load_windows_from_dir(d: Path) -> np.ndarray:
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(d)
    # Prefer windows.npz with key X
    npz = d / "windows.npz"
    if npz.exists():
        data = np.load(npz)
        X = data.get("X") if isinstance(data, np.lib.npyio.NpzFile) else None
        if X is None:
            X = data.get("arr_0")
        if X is None:
            raise ValueError(f"{npz} missing 'X' or 'arr_0'")
        return np.asarray(X, dtype=np.float32)
    # Fallback: any *.npy in the directory
    npys = list(d.glob("*.npy"))
    if npys:
        X = np.load(npys[0])
        return np.asarray(X, dtype=np.float32)
    raise FileNotFoundError(f"No windows.npz or *.npy found in {d}")


def _zscore_per_channel(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    return (x - mean) / np.clip(std, 1e-8, None)


def _windows_from_csv(csv: Path, window_len: int, stride: int) -> np.ndarray:
    if pd is None:
        raise SystemExit("pandas required for CSV mode: pip install pandas")
    df = pd.read_csv(csv)
    # Keep numeric columns only
    num_df = df.select_dtypes(include=["number"]).copy()
    if num_df.shape[1] == 0:
        raise ValueError("CSV must contain numeric columns")
    data = num_df.to_numpy(dtype=np.float32)
    # Channels = columns, treat rows as time
    C = data.shape[1]
    L = window_len
    windows = []
    for start in range(0, data.shape[0] - L + 1, stride):
        sl = data[start : start + L]  # [L, C]
        win = sl.T  # [C, L]
        windows.append(win)
    if not windows:
        raise ValueError("CSV too short for given window_len/stride")
    X = np.stack(windows, axis=0).astype(np.float32)
    return _zscore_per_channel(X)


def _load_checkpoint(ckpt: Path, in_channels: int, window_len: int = 200) -> torch.nn.Module:
    model = build_model_from_config(window_len=window_len, in_channels=in_channels, num_heads=4)
    if ckpt.exists():
        obj = torch.load(ckpt, map_location="cpu")
        state = obj.get("state_dict", obj)
        try:
            model.load_state_dict(state)
        except Exception:
            # If shape mismatch, try to load with different in_channels if available
            meta_in = int(obj.get("in_channels", in_channels))
            model = build_model_from_config(window_len=window_len, in_channels=meta_in, num_heads=4)
            model.load_state_dict(state)
    return model


def main() -> None:
    cfg = load_config()

    threads = int(cfg.get("system", "threads", default=0) or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    show_progress: bool = bool(cfg.get("system", "show_progress", default=True))

    # Inputs
    input_dir = cfg.get("inference", "input_dir", default=None)
    price_csv = cfg.get("inference", "price_csv", default=None)
    window_len = int(cfg.get("inference", "window_len", default=128) or 128)
    stride = int(cfg.get("inference", "stride", default=8) or 8)

    enable_gradcam: bool = bool(cfg.get("inference", "enable_gradcam", default=True))
    output_jsonl = Path(str(cfg.get("inference", "output_jsonl", default="models/cnn_bullflag/example_cnn_outputs.jsonl")))

    # Resolve windows
    X: Optional[np.ndarray] = None
    if input_dir:
        X = _load_windows_from_dir(Path(str(input_dir)))
    elif price_csv:
        X = _windows_from_csv(Path(str(price_csv)), window_len=window_len, stride=stride)
    else:
        # Default to loading from dataset.npz (train_x)
        dataset_path = Path(str(cfg.get("data", "train_x", default="")))
        if dataset_path.exists() and dataset_path.suffix == ".npz":
            data = np.load(dataset_path)
            X = data.get("X")
            if X is None:
                X = data.get("arr_0")
            if X is None:
                raise ValueError(f"Dataset {dataset_path} missing 'X' or 'arr_0'")
            X = np.asarray(X, dtype=np.float32)
        else:
            raise SystemExit("Set [inference].input_dir, [inference].price_csv, or ensure [data].train_x points to valid dataset.npz")

    if X.ndim != 3:
        raise ValueError(f"X must be [N,C,L]; got {X.shape}")

    # Load model
    ckpt = Path(str(cfg.get("checkpoint", "model_path", default="models/cnn_bullflag/checkpoint.pt")))
    model = _load_checkpoint(ckpt, in_channels=X.shape[1], window_len=X.shape[2])

    # Build inference runner
    runner = BullFlagCNNInfer(model=model, enable_gradcam=enable_gradcam, gradcam_target_layer="conv2")

    # Iterate and write JSONL
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    it = enumerate(X)
    if show_progress and tqdm is not None:
        it = tqdm(it, total=X.shape[0], desc="Infer", unit="win")  # type: ignore
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for idx, win in it:  # type: ignore
            meta = WindowMeta(
                window_id=f"window_{idx:04d}",
                symbol=str(cfg.get("inference", "symbol", default="DEMO")),
                timeframe=str(cfg.get("inference", "timeframe", default="5m")),
                start_ts=idx,
                end_ts=idx + (win.shape[-1] - 1),
            )
            row = runner.infer_with_schema(win, meta)
            fh.write(json.dumps(row))
            fh.write("\n")

    print(f"Wrote JSONL outputs to {output_jsonl}")


if __name__ == "__main__":
    main()
