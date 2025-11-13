#!/usr/bin/env python3
"""
Download additional BTCUSDT 30m data and extend the existing dataset.
Supports multiple sources: CryptoDataDownload, Binance API, or synthetic generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._setup_path import add_project_root

add_project_root()

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import requests
except ImportError:
    requests = None

from scripts.download_and_extract_windows import (
    compute_ta_features,
    build_sliding_windows,
    zscore_normalize,
)


def download_binance_klines(symbol: str = "BTCUSDT", interval: str = "30m", limit: int = 1000) -> Optional[np.ndarray]:
    """Download klines from Binance public API."""
    if requests is None:
        print("requests library not available, skipping Binance download")
        return None
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)  # Binance max is 1000 per request
    }
    
    try:
        print(f"Downloading {limit} candles from Binance ({symbol}, {interval})...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Binance format: [openTime, open, high, low, close, volume, closeTime, ...]
        # We need: open, high, low, close, volume (columns 1-5)
        ohlcv = []
        for kline in data:
            ohlcv.append([
                float(kline[1]),  # open
                float(kline[2]),  # high
                float(kline[3]),  # low
                float(kline[4]),  # close
                float(kline[5]),  # volume
            ])
        
        # Binance returns newest first, so reverse to oldest first
        ohlcv_array = np.array(ohlcv[::-1])
        print(f"Downloaded {len(ohlcv_array)} candles")
        return ohlcv_array
        
    except Exception as e:
        print(f"Binance download failed: {e}")
        return None


def download_cryptodatadownload(symbol: str = "BTCUSDT", timeframe: str = "30m") -> Optional[np.ndarray]:
    """Download from CryptoDataDownload (if available)."""
    if requests is None or pd is None:
        return None
    
    # CryptoDataDownload URL format
    url = f"https://www.cryptodatadownload.com/cdd/Binance_{symbol}_{timeframe}.csv"
    
    try:
        print(f"Attempting to download from CryptoDataDownload...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save temporarily
        temp_path = Path("data/temp_cdd.csv")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(response.content)
        
        # Parse CSV
        df = pd.read_csv(temp_path, skiprows=1)  # Skip header row
        required_cols = ["open", "high", "low", "close", "Volume USDT"]
        if all(col in df.columns for col in required_cols):
            data = df[required_cols].astype(float).values
            # CDD format is newest first, reverse
            data = np.flip(data, axis=0)
            temp_path.unlink()  # Clean up
            print(f"Downloaded {len(data)} candles from CryptoDataDownload")
            return data
        else:
            temp_path.unlink()
            return None
            
    except Exception as e:
        print(f"CryptoDataDownload download failed: {e}")
        return None


def generate_extended_synthetic(num_rows: int = 20000, seed: Optional[int] = None) -> np.ndarray:
    """Generate extended synthetic BTCUSDT 30m OHLCV data."""
    if seed is not None:
        np.random.seed(seed)
    base_price = 50000.0
    prices = [base_price]
    
    for _ in range(num_rows - 1):
        # Random walk with volatility
        change = np.random.normal(0, 0.01)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))
    
    data = []
    for i in range(num_rows):
        open_p = prices[i]
        close_p = prices[min(i+1, len(prices)-1)]
        high_p = max(open_p, close_p) * (1 + np.random.uniform(0, 0.005))
        low_p = min(open_p, close_p) * (1 - np.random.uniform(0, 0.005))
        volume = np.random.uniform(100, 1000)
        data.append([open_p, high_p, low_p, close_p, volume])
    
    return np.array(data)


def load_existing_dataset(dataset_path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load existing dataset and return X, regimes."""
    if not dataset_path.exists():
        return None, None
    
    data = np.load(dataset_path, allow_pickle=False)
    X = data.get("X")
    regimes = data.get("regimes")
    return X, regimes


def extend_dataset(
    existing_X: np.ndarray,
    new_ohlcv: np.ndarray,
    window_len: int = 200,
    stride: int = 1,
    normalize_together: bool = True,
) -> np.ndarray:
    """Extend existing dataset with new OHLCV data."""
    # Compute TA features for new data
    ta_data = compute_ta_features(new_ohlcv)
    
    # Normalize new data (can normalize together or separately)
    if normalize_together and existing_X is not None:
        # Combine for normalization stats
        # Extract original features from existing windows (approximate)
        # For simplicity, normalize separately but this could be improved
        ta_data_norm = zscore_normalize(ta_data)
    else:
        ta_data_norm = zscore_normalize(ta_data)
    
    # Build windows from new data
    new_X = build_sliding_windows(ta_data_norm, window_len=window_len, stride=stride)
    
    # Combine with existing
    if existing_X is not None:
        # Re-normalize existing data to match new normalization
        # For now, just concatenate (in production, you'd want to normalize together)
        combined_X = np.concatenate([existing_X, new_X], axis=0)
    else:
        combined_X = new_X
    
    return combined_X


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather additional BTCUSDT 30m data")
    parser.add_argument("--source", type=str, default="binance", choices=["binance", "cdd", "synthetic"],
                       help="Data source")
    parser.add_argument("--limit", type=int, default=5000, help="Number of candles to download/generate")
    parser.add_argument("--dataset", type=str, default="data/dataset.npz", help="Existing dataset path")
    parser.add_argument("--out", type=str, default="data/dataset.npz", help="Output dataset path")
    parser.add_argument("--window-len", type=int, default=200, help="Window length")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sliding windows")
    parser.add_argument("--append", action="store_true", help="Append to existing dataset")
    args = parser.parse_args()
    
    # Load existing dataset
    existing_X = None
    existing_regimes = None
    if args.append:
        dataset_path = Path(args.dataset)
        existing_X, existing_regimes = load_existing_dataset(dataset_path)
        if existing_X is not None:
            print(f"Loaded existing dataset: {existing_X.shape}")
    
    # Gather new data
    new_ohlcv = None
    if args.source == "binance":
        new_ohlcv = download_binance_klines(limit=args.limit)
    elif args.source == "cdd":
        new_ohlcv = download_cryptodatadownload()
    elif args.source == "synthetic":
        print(f"Generating {args.limit} synthetic candles...")
        new_ohlcv = generate_extended_synthetic(num_rows=args.limit, seed=None)
    
    if new_ohlcv is None:
        print("Failed to gather new data, falling back to synthetic")
        new_ohlcv = generate_extended_synthetic(num_rows=args.limit, seed=None)
    
    print(f"New OHLCV data shape: {new_ohlcv.shape}")
    
    # Extend dataset
    if args.append and existing_X is not None:
        print("Extending existing dataset...")
        combined_X = extend_dataset(existing_X, new_ohlcv, window_len=args.window_len, stride=args.stride)
    else:
        print("Creating new dataset...")
        ta_data = compute_ta_features(new_ohlcv)
        ta_data_norm = zscore_normalize(ta_data)
        combined_X = build_sliding_windows(ta_data_norm, window_len=args.window_len, stride=args.stride)
    
    print(f"Final dataset shape: {combined_X.shape}")
    
    # Create dummy labels
    N = combined_X.shape[0]
    y = np.zeros((N, 4), dtype=np.float32)
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"X": combined_X, "y": y}
    if existing_regimes is not None:
        # Extend regimes with None/empty strings
        new_regimes = np.array([""] * (N - len(existing_regimes)), dtype=object)
        combined_regimes = np.concatenate([existing_regimes, new_regimes])
        save_kwargs["regimes"] = combined_regimes
    
    np.savez(out_path, **save_kwargs)
    print(f"Saved extended dataset to {out_path}")
    print(f"Total windows: {N}")


if __name__ == "__main__":
    main()

