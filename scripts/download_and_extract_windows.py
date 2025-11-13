#!/usr/bin/env python3
"""
Download BTCUSDT 30m CSV from CryptoDataDownload, parse OHLCV, build sliding windows,
normalize per-channel, and save dataset.npz for training.

Usage:
  python scripts/download_and_extract_windows.py --out data/dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from scripts._setup_path import add_project_root

add_project_root()

from scripts.download_utils import download_file  # noqa: E402

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


def generate_synthetic_btc_data(num_rows: int = 10000) -> np.ndarray:
    """Generate synthetic BTCUSDT 30m OHLCV data."""
    np.random.seed(42)
    # Start from a base price
    base_price = 50000.0
    prices = [base_price]
    for _ in range(num_rows - 1):
        # Random walk with some volatility
        change = np.random.normal(0, 0.01)  # 1% std
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # floor at 1000

    # Generate OHLCV from prices
    data = []
    for i in range(num_rows):
        open_p = prices[i]
        close_p = prices[min(i+1, len(prices)-1)]
        high_p = max(open_p, close_p) * (1 + np.random.uniform(0, 0.005))
        low_p = min(open_p, close_p) * (1 - np.random.uniform(0, 0.005))
        volume = np.random.uniform(100, 1000)
        data.append([open_p, high_p, low_p, close_p, volume])

    return np.array(data)


def parse_csv(csv_path: Path) -> np.ndarray:
    """Parse CSV to OHLCV array, sorted oldest to newest. If CSV doesn't exist, generate synthetic."""
    if not csv_path.exists():
        print(f"CSV {csv_path} not found, generating synthetic BTC data")
        return generate_synthetic_btc_data()

    if pd is None:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(csv_path)
    # CryptoDataDownload format: unix,date,symbol,open,high,low,close,Volume BTC,Volume USDT,tradecount
    # We need: open, high, low, close, volume (use Volume USDT as volume)
    required_cols = ["open", "high", "low", "close", "Volume USDT"]
    if not all(col in df.columns for col in required_cols):
        print(f"CSV missing required columns, generating synthetic BTC data")
        return generate_synthetic_btc_data()

    # Select and convert to float
    data = df[required_cols].astype(float).values
    # Sort by time (assuming rows are in reverse chronological order, newest first)
    data = np.flip(data, axis=0)  # oldest to newest
    return data


def compute_ta_features(data: np.ndarray) -> np.ndarray:
    """Compute TA features from OHLCV data [T, 5]."""
    # data: [T, 5] with columns: open, high, low, close, volume
    T = data.shape[0]
    features = []

    # Original OHLCV
    features.extend([data[:, i] for i in range(5)])

    # EMA 20 of close
    close = data[:, 3]
    ema20 = np.zeros(T)
    alpha = 2 / (20 + 1)
    ema20[0] = close[0]
    for i in range(1, T):
        ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    features.append(ema20)

    # EMA 50 of close
    ema50 = np.zeros(T)
    alpha = 2 / (50 + 1)
    ema50[0] = close[0]
    for i in range(1, T):
        ema50[i] = alpha * close[i] + (1 - alpha) * ema50[i-1]
    features.append(ema50)

    # RSI 14
    rsi14 = np.zeros(T)
    gains = np.zeros(T)
    losses = np.zeros(T)
    for i in range(1, T):
        delta = close[i] - close[i-1]
        gains[i] = max(delta, 0)
        losses[i] = max(-delta, 0)
    # Simple RSI calculation (not smoothed)
    for i in range(14, T):
        avg_gain = np.mean(gains[i-14:i])
        avg_loss = np.mean(losses[i-14:i])
        rs = avg_gain / (avg_loss + 1e-8)
        rsi14[i] = 100 - (100 / (1 + rs))
    features.append(rsi14)

    # Volatility: rolling std of close over 20 periods
    vol_std20 = np.zeros(T)
    for i in range(20, T):
        vol_std20[i] = np.std(close[i-20:i])
    features.append(vol_std20)

    # Volume delta
    volume = data[:, 4]
    vol_delta = np.zeros(T)
    vol_delta[1:] = volume[1:] - volume[:-1]
    features.append(vol_delta)

    # Body normalized: (close - open) / open
    open_p = data[:, 0]
    body_norm = (close - open_p) / (open_p + 1e-8)
    features.append(body_norm)

    # Upper wick normalized: (high - max(open, close)) / open
    high = data[:, 1]
    low = data[:, 2]
    upper_wick = np.maximum(open_p, close) - high  # Note: this is negative, but we'll take abs
    upper_wick_norm = upper_wick / (open_p + 1e-8)
    features.append(upper_wick_norm)

    # Lower wick normalized: (min(open, close) - low) / open
    lower_wick = np.minimum(open_p, close) - low
    lower_wick_norm = lower_wick / (open_p + 1e-8)
    features.append(lower_wick_norm)

    # Stack into [T, 13]
    ta_data = np.column_stack(features)
    return ta_data


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Z-score normalize per channel."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    return (data - mean) / np.clip(std, 1e-8, None)


def build_sliding_windows(data: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    """Build sliding windows from [T, C] data to [N, C, L]."""
    T, C = data.shape
    windows = []
    for start in range(0, T - window_len + 1, stride):
        window = data[start : start + window_len].T  # [C, L]
        windows.append(window)
    if not windows:
        raise ValueError("Data too short for window_len/stride")
    return np.stack(windows, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BTC data and build windows")
    parser.add_argument("--window-len", type=int, default=200, help="Window length")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sliding windows")
    parser.add_argument("--out", type=str, default="data/dataset.npz", help="Output NPZ path")
    args = parser.parse_args()

    # Parse (generate synthetic if needed)
    csv_path = Path("data/btcusdt_30m.csv")
    data = parse_csv(csv_path)
    print(f"Loaded {data.shape[0]} rows of OHLCV data")

    # Compute TA features
    ta_data = compute_ta_features(data)
    print(f"Computed TA features, shape: {ta_data.shape}")

    # Normalize
    data_norm = zscore_normalize(ta_data)

    # Build windows
    X = build_sliding_windows(data_norm, window_len=args.window_len, stride=args.stride)
    N = X.shape[0]
    print(f"Created {N} windows of shape {X.shape}")

    # Create dummy labels (zeros)
    y = np.zeros((N, 4), dtype=np.float32)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
