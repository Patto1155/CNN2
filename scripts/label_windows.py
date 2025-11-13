#!/usr/bin/env python3
"""
Deterministic multi-head labeler for the BullFlag CNN dataset.

Semantics implemented exactly as specified:
- flag: bull flag consolidation in progress.
- breakout: confirmed breakout 10-40 bars after a flag.
- retest: pullback to breakout level within +/-0.5 ATR followed by momentum regain.
- continuation: rally following a successful retest.

Outputs data/dataset_labeled.npz with X, y, and optional regimes.
Also updates config.toml [data] paths, prints label stats, and saves preview plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python <3.11
    import tomli as tomllib  # type: ignore[assignment]

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._setup_path import add_project_root

add_project_root()

# Use non-interactive backend before importing pyplot.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    tqdm = None


CHANNELS = {
    "open": 0,
    "high": 1,
    "low": 2,
    "close": 3,
    "volume": 4,
    "ema20": 5,
    "ema50": 6,
}

LABEL_NAMES = ("flag", "breakout", "retest", "continuation")
RECENT_EVENT_LOOKBACK = 20


@dataclass(frozen=True)
class LabelerParams:
    pattern_zone: int = 80
    k_impulse_min: int = 15
    k_impulse_max: int = 40
    impulse_min_move_pct: float = 0.05
    impulse_vol_mult: float = 1.5
    flag_min_bars: int = 15
    flag_max_bars: int = 60
    flag_max_range_pct: float = 0.04
    flag_std_rel_max: float = 0.7
    ema_flat_slope_max: float = 0.001
    flag_vol_rel_max: float = 0.8
    breakout_min_pct: float = 0.01
    breakout_max_lookahead: int = 40
    breakout_vol_mult: float = 1.5
    retest_window_pullback: int = 25
    retest_window_reclaim: int = 25
    retest_atr_tolerance: float = 0.5
    reclaim_min_pct: float = 0.0
    continuation_window: int = 40
    continuation_min_pct: float = 0.02
    min_head_prevalence: float = 0.01
    max_head_prevalence: float = 0.05


@dataclass
class LabelDiagnostics:
    """Captures key event indices for bull-flag detection."""

    t_flag_end: Optional[int] = None
    t_breakout: Optional[int] = None
    t_retest: Optional[int] = None
    t_continuation: Optional[int] = None


DEFAULT_PARAMS = LabelerParams()


def load_labeler_params(config_path: Path = Path("config.toml")) -> LabelerParams:
    """Load labeler hyperparameters from config.toml."""

    if not config_path.exists():
        return DEFAULT_PARAMS
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)
    section = data.get("labeler", {}) or {}
    return LabelerParams(
        pattern_zone=int(section.get("pattern_zone", DEFAULT_PARAMS.pattern_zone)),
        k_impulse_min=int(section.get("k_impulse_min", DEFAULT_PARAMS.k_impulse_min)),
        k_impulse_max=int(section.get("k_impulse_max", DEFAULT_PARAMS.k_impulse_max)),
        impulse_min_move_pct=float(section.get("impulse_min_move_pct", DEFAULT_PARAMS.impulse_min_move_pct)),
        impulse_vol_mult=float(section.get("impulse_vol_mult", DEFAULT_PARAMS.impulse_vol_mult)),
        flag_min_bars=int(section.get("flag_min_bars", DEFAULT_PARAMS.flag_min_bars)),
        flag_max_bars=int(section.get("flag_max_bars", DEFAULT_PARAMS.flag_max_bars)),
        flag_max_range_pct=float(section.get("flag_max_range_pct", DEFAULT_PARAMS.flag_max_range_pct)),
        flag_std_rel_max=float(section.get("flag_std_rel_max", DEFAULT_PARAMS.flag_std_rel_max)),
        ema_flat_slope_max=float(section.get("ema_flat_slope_max", DEFAULT_PARAMS.ema_flat_slope_max)),
        flag_vol_rel_max=float(section.get("flag_vol_rel_max", DEFAULT_PARAMS.flag_vol_rel_max)),
        breakout_min_pct=float(section.get("breakout_min_pct", DEFAULT_PARAMS.breakout_min_pct)),
        breakout_max_lookahead=int(section.get("breakout_max_lookahead", DEFAULT_PARAMS.breakout_max_lookahead)),
        breakout_vol_mult=float(section.get("breakout_vol_mult", DEFAULT_PARAMS.breakout_vol_mult)),
        retest_window_pullback=int(section.get("retest_window_pullback", DEFAULT_PARAMS.retest_window_pullback)),
        retest_window_reclaim=int(section.get("retest_window_reclaim", DEFAULT_PARAMS.retest_window_reclaim)),
        retest_atr_tolerance=float(section.get("retest_atr_tolerance", DEFAULT_PARAMS.retest_atr_tolerance)),
        reclaim_min_pct=float(section.get("reclaim_min_pct", DEFAULT_PARAMS.reclaim_min_pct)),
        continuation_window=int(section.get("continuation_window", DEFAULT_PARAMS.continuation_window)),
        continuation_min_pct=float(section.get("continuation_min_pct", DEFAULT_PARAMS.continuation_min_pct)),
        min_head_prevalence=float(section.get("min_head_prevalence", DEFAULT_PARAMS.min_head_prevalence)),
        max_head_prevalence=float(section.get("max_head_prevalence", DEFAULT_PARAMS.max_head_prevalence)),
    )


def load_system_show_progress(config_path: Path = Path("config.toml")) -> bool:
    """Return whether console progress reporting should be enabled."""

    if not config_path.exists():
        return False
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)
    section = data.get("system", {}) or {}
    return bool(section.get("show_progress", False))


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return arrays


def linear_slope(values: np.ndarray) -> float:
    n = values.size
    if n <= 1:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    sum_x = np.sum(x)
    sum_y = np.sum(values)
    sum_x2 = np.dot(x, x)
    sum_xy = np.dot(x, values)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(series, np.nan, dtype=np.float64)
    if series.size < window:
        return result
    cumsum = np.cumsum(series, dtype=np.float64)
    prev = np.concatenate(([0.0], cumsum[:-window]))
    result[window - 1 :] = (cumsum[window - 1 :] - prev) / window
    return result


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - np.concatenate(([close[0]], close[:-1]))),
            np.abs(low - np.concatenate(([close[0]], close[:-1]))),
        ]
    )
    atr = np.full_like(close, np.nan, dtype=np.float64)
    if close.size < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, close.size):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def rate_of_change(close: np.ndarray, period: int = 10) -> np.ndarray:
    roc = np.full_like(close, np.nan, dtype=np.float64)
    if close.size <= period:
        return roc
    denom = close[:-period]
    denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
    roc[period:] = (close[period:] - close[:-period]) / denom
    return roc


def _flag_condition(
    idx: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    atr: np.ndarray,
    params: LabelerParams,
) -> bool:
    """Heuristic for detecting a flag consolidation at index idx."""

    if idx < 60 or idx >= close.size:
        return False

    uptrend_slice = slice(idx - 59, idx + 1)
    uptrend_ratio = np.mean(ema20[uptrend_slice] > ema50[uptrend_slice])
    if uptrend_ratio < params.flag_min_uptrend_fraction:
        return False

    atr_recent = atr[idx - 19 : idx + 1]
    atr_previous = atr[idx - 39 : idx - 19]
    if atr_recent.size < 20 or atr_previous.size < 20:
        return False
    if np.isnan(atr_recent).any() or np.isnan(atr_previous).any():
        return False
    prev_mean = np.mean(atr_previous)
    if prev_mean <= 0:
        return False
    atr_drop = (prev_mean - np.mean(atr_recent)) / prev_mean
    if atr_drop < params.flag_atr_drop_pct:
        return False

    price_window = close[idx - 19 : idx + 1]
    mean_price = np.mean(price_window)
    if mean_price <= 0:
        return False
    std_ratio = np.std(price_window) / mean_price
    if std_ratio >= params.flag_std_max:
        return False

    high_window = high[idx - 19 : idx + 1]
    low_window = low[idx - 19 : idx + 1]
    if linear_slope(high_window) >= 0 or linear_slope(low_window) >= 0:
        return False

    return True


def detect_impulse(
    close: np.ndarray,
    volume: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    params: LabelerParams,
) -> Optional[Dict[str, Any]]:
    """Detect a qualifying impulse leg. Indices are inclusive."""

    L = close.size
    if L < params.k_impulse_min:
        return None

    required_tail = params.flag_min_bars + params.breakout_max_lookahead
    latest_end = L - required_tail - 1
    if latest_end < params.k_impulse_min - 1:
        latest_end = L - 2

    best_candidate: Optional[Dict[str, Any]] = None
    for end_idx in range(latest_end, params.k_impulse_min - 2, -1):
        max_length = min(params.k_impulse_max, end_idx + 1)
        for length in range(max_length, params.k_impulse_min - 1, -1):
            start_idx = end_idx - length + 1
            if start_idx < 0:
                continue

            segment_close = close[start_idx : end_idx + 1]
            if np.isnan(segment_close).any():
                continue
            move_pct = float(segment_close[-1] / segment_close[0] - 1.0)
            if move_pct < params.impulse_min_move_pct:
                continue
            if linear_slope(segment_close) <= 0:
                continue
            if ema20[end_idx] <= ema50[end_idx]:
                continue

            baseline_start = max(0, start_idx - length)
            baseline = volume[baseline_start:start_idx]
            if baseline.size < max(1, length // 2):
                continue
            baseline_mean = float(np.mean(baseline))
            if not np.isfinite(baseline_mean) or baseline_mean <= 0:
                continue

            segment_volume = volume[start_idx : end_idx + 1]
            if np.isnan(segment_volume).any():
                continue
            segment_mean = float(np.mean(segment_volume))
            if segment_mean < params.impulse_vol_mult * baseline_mean:
                continue

            best_candidate = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "length": length,
                "move_pct": move_pct,
                "close_start": float(segment_close[0]),
                "close_end": float(segment_close[-1]),
                "volume_mean": segment_mean,
                "price_std": float(np.std(segment_close)),
            }
            break
        if best_candidate is not None:
            break

    return best_candidate


def detect_flag(
    close: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    volume: np.ndarray,
    impulse: Dict[str, Any],
    params: LabelerParams,
) -> Optional[Dict[str, Any]]:
    """Detect bull flag consolidation after impulse."""

    L = close.size
    if L < params.flag_min_bars:
        return None

    pattern_start = max(0, L - params.pattern_zone)
    min_start = max(pattern_start, impulse["end_idx"] + 1)
    if min_start + params.flag_min_bars - 1 >= L:
        return None

    max_end = min(L - 2, pattern_start + params.pattern_zone - 1)
    max_end = min(max_end, L - params.breakout_max_lookahead - 1)
    if max_end < min_start + params.flag_min_bars - 1:
        return None

    impulse_std = impulse.get("price_std", float(np.std(close[impulse["start_idx"] : impulse["end_idx"] + 1])))
    impulse_vol_mean = impulse["volume_mean"]
    best_candidate: Optional[Dict[str, Any]] = None

    for end_idx in range(max_end, min_start + params.flag_min_bars - 2, -1):
        for length in range(params.flag_max_bars, params.flag_min_bars - 1, -1):
            start_idx = end_idx - length + 1
            if start_idx < min_start:
                continue
            if end_idx >= L:
                continue

            flag_close = close[start_idx : end_idx + 1]
            flag_ema20 = ema20[start_idx : end_idx + 1]
            flag_volume = volume[start_idx : end_idx + 1]
            if np.isnan(flag_close).any() or np.isnan(flag_ema20).any():
                continue

            flag_high = float(np.max(flag_close))
            flag_low = float(np.min(flag_close))
            mid_price = (flag_high + flag_low) / 2.0
            if mid_price <= 0:
                continue
            range_pct = (flag_high - flag_low) / mid_price
            if range_pct > params.flag_max_range_pct:
                continue

            if impulse_std <= 0:
                continue
            flag_std = float(np.std(flag_close))
            if flag_std > params.flag_std_rel_max * impulse_std:
                continue

            ema_slope = linear_slope(flag_ema20)
            if abs(ema_slope) > params.ema_flat_slope_max * mid_price:
                continue

            flag_vol_mean = float(np.mean(flag_volume))
            if flag_vol_mean > params.flag_vol_rel_max * impulse_vol_mean:
                continue

            if ema20[end_idx] < ema50[end_idx]:
                continue

            best_candidate = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "high": flag_high,
                "low": flag_low,
                "volume_mean": flag_vol_mean,
            }
            break
        if best_candidate is not None:
            break

    return best_candidate


def detect_breakout(
    close: np.ndarray,
    high: np.ndarray,
    volume: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    flag: Dict[str, Any],
    params: LabelerParams,
) -> Optional[int]:
    """Detect breakout above flag high."""
    L = close.size
    pattern_start = max(0, L - params.pattern_zone)

    flag_end = flag["end_idx"]
    flag_high = flag["high"]
    flag_vol_mean = flag["volume_mean"]

    # Look for breakout within lookahead window
    for t in range(flag_end + 1, min(flag_end + params.breakout_max_lookahead + 1, L)):
        # Must be in pattern zone
        if t < pattern_start:
            continue

        # Price breakout
        breakout_level = flag_high * (1 + params.breakout_min_pct)
        if close[t] < breakout_level:
            continue
        if high[t] < breakout_level:
            continue

        # EMA alignment
        if ema20[t] <= ema50[t]:
            continue

        # Volume spike
        if volume[t] < params.breakout_vol_mult * flag_vol_mean:
            continue

        return t

    return None


def detect_retest(
    close: np.ndarray,
    vol_std20: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    flag: Dict[str, Any],
    breakout_idx: int,
    params: LabelerParams,
) -> Optional[int]:
    """Detect retest with reclaim."""

    L = close.size
    pattern_start = max(0, L - params.pattern_zone)
    breakout_level = close[breakout_idx]
    flag_high = flag["high"]
    flag_low = flag["low"]

    atr_window = vol_std20[max(0, breakout_idx - 10) : min(L, breakout_idx + 11)]
    atr_proxy = float(np.nanmedian(atr_window)) if atr_window.size else np.nan
    if not np.isfinite(atr_proxy) or atr_proxy <= 0:
        atr_proxy = float(np.nanmean(vol_std20))
    if not np.isfinite(atr_proxy) or atr_proxy <= 0:
        atr_proxy = abs(breakout_level - flag_low)
    if atr_proxy <= 0:
        return None

    pullback_end = min(L - 1, breakout_idx + params.retest_window_pullback)
    tolerance = params.retest_atr_tolerance * atr_proxy
    for t_pull in range(breakout_idx + 1, pullback_end + 1):
        if t_pull < pattern_start:
            continue
        price = close[t_pull]
        if price > breakout_level:
            continue
        if price < flag_low:
            continue
        if abs(price - breakout_level) > tolerance:
            continue

        reclaim_end = min(L - 1, t_pull + params.retest_window_reclaim)
        reclaim_target = max(breakout_level * (1 + params.reclaim_min_pct), flag_high)
        for t_reclaim in range(t_pull + 1, reclaim_end + 1):
            if t_reclaim < pattern_start:
                continue
            if close[t_reclaim] < reclaim_target:
                continue
            if ema20[t_reclaim] <= ema50[t_reclaim]:
                continue
            if ema20[t_reclaim] <= ema20[max(t_reclaim - 1, 0)]:
                continue
            recent = ema20[max(0, t_reclaim - 3) : t_reclaim + 1]
            if recent.size >= 2 and linear_slope(recent) <= 0:
                continue
            return t_reclaim

    return None


def detect_continuation(
    close: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    breakout_idx: int,
    retest_idx: int,
    params: LabelerParams,
) -> Optional[int]:
    """Detect fresh leg up after retest."""
    L = close.size
    pattern_start = max(0, L - params.pattern_zone)

    breakout_level = close[breakout_idx]

    # Look for continuation within window
    for t in range(retest_idx + 1, min(retest_idx + params.continuation_window + 1, L)):
        # Must be in pattern zone
        if t < pattern_start:
            continue

        # Price continuation
        if close[t] < breakout_level * (1 + params.continuation_min_pct):
            continue

        # Still in uptrend
        if ema20[t] <= ema20[retest_idx]:
            continue
        if ema20[t] <= ema50[t]:
            continue
        if ema20[t] <= ema20[max(0, t - 1)]:
            continue

        return t

    return None


def label_single_window(
    window: np.ndarray, params: Optional[LabelerParams] = None
) -> Tuple[np.ndarray, LabelDiagnostics]:
    params = params or DEFAULT_PARAMS

    close = window[CHANNELS["close"]].astype(np.float64)
    volume = window[CHANNELS["volume"]].astype(np.float64)
    ema20 = window[CHANNELS["ema20"]].astype(np.float64)
    ema50 = window[CHANNELS["ema50"]].astype(np.float64)
    high = window[CHANNELS["high"]].astype(np.float64)
    vol_std20 = window[8].astype(np.float64)  # vol_std20 / atr proxy channel

    diag = LabelDiagnostics()
    label_vec = np.zeros(4, dtype=np.int32)

    impulse = detect_impulse(close, volume, ema20, ema50, params)
    if impulse is None:
        return label_vec, diag

    flag = detect_flag(close, ema20, ema50, volume, impulse, params)
    if flag is None:
        return label_vec, diag

    diag.t_flag_end = flag["end_idx"]
    label_vec[0] = 1

    breakout_idx = detect_breakout(close, high, volume, ema20, ema50, flag, params)
    if breakout_idx is None:
        return label_vec, diag

    diag.t_breakout = breakout_idx
    label_vec[1] = 1

    retest_idx = detect_retest(close, vol_std20, ema20, ema50, flag, breakout_idx, params)
    if retest_idx is None:
        return label_vec, diag

    diag.t_retest = retest_idx
    label_vec[2] = 1

    continuation_idx = detect_continuation(close, ema20, ema50, breakout_idx, retest_idx, params)
    if continuation_idx is None:
        return label_vec, diag

    diag.t_continuation = continuation_idx
    label_vec[3] = 1
    return label_vec, diag


def label_dataset(
    X: np.ndarray,
    *,
    params: Optional[LabelerParams] = None,
    return_diagnostics: bool = False,
    show_progress: bool = False,
) -> Tuple[np.ndarray, Optional[List[LabelDiagnostics]]]:
    params = params or DEFAULT_PARAMS
    total = int(X.shape[0])
    labels = np.zeros((total, len(LABEL_NAMES)), dtype=np.int32)
    diagnostics: List[LabelDiagnostics] = []
    progress_bar = None
    log_interval = None
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Labeling windows", unit="win")
        else:
            log_interval = max(1, total // 20)
    try:
        for idx in range(total):
            window = X[idx]
        label_vec, diag = label_single_window(window, params=params)
        labels[idx] = label_vec
        if return_diagnostics:
            diagnostics.append(diag)
        if show_progress and progress_bar is not None:
            progress_bar.update(1)
        elif show_progress and log_interval is not None:
            if (idx + 1) % log_interval == 0 or idx + 1 == total:
                print(f"Labeling windows: {idx + 1}/{total}")
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return labels, (diagnostics if return_diagnostics else None)


def _plot_sample_windows(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    out_path: Path,
) -> None:
    cols = 1
    rows = sample_indices.size
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), sharex=False)
    if rows == 1:
        axes = [axes]
    for ax, idx in zip(axes, sample_indices):
        window = X[idx]
        close = window[CHANNELS["close"]]
        ema20 = window[CHANNELS["ema20"]]
        ema50 = window[CHANNELS["ema50"]]
        ax.plot(close, label="close", linewidth=1.2)
        ax.plot(ema20, label="ema20", linewidth=0.9)
        ax.plot(ema50, label="ema50", linewidth=0.9)
        label_text = ", ".join(
            f"{name}={y[idx, i]}" for i, name in enumerate(LABEL_NAMES)
        )
        ax.set_title(f"Window {idx} | {label_text}")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _min_gap(values: List[int]) -> Optional[int]:
    return min(values) if values else None


def summarize_labels(
    labels: np.ndarray, diagnostics: Optional[List[LabelDiagnostics]], params: Optional[LabelerParams] = None
) -> Dict[str, Any]:
    params = params or DEFAULT_PARAMS
    total = int(labels.shape[0])
    counts = labels.sum(axis=0).astype(int)
    percentages = (counts / total * 100) if total > 0 else np.zeros_like(counts, dtype=float)

    diag_list = diagnostics or []
    flag_breakout_gaps: List[int] = []
    breakout_retest_gaps: List[int] = []
    retest_continuation_gaps: List[int] = []
    flags_to_breakouts = 0
    breakouts_to_retests = 0
    retests_to_continuations = 0

    for diag in diag_list:
        if diag.t_flag_end is not None and diag.t_breakout is not None:
            flags_to_breakouts += 1
            flag_breakout_gaps.append(max(0, diag.t_breakout - diag.t_flag_end))
        if diag.t_breakout is not None and diag.t_retest is not None:
            breakouts_to_retests += 1
            breakout_retest_gaps.append(max(0, diag.t_retest - diag.t_breakout))
        if diag.t_retest is not None and diag.t_continuation is not None:
            retests_to_continuations += 1
            retest_continuation_gaps.append(max(0, diag.t_continuation - diag.t_retest))

    summary = {
        "total": total,
        "counts": counts.tolist(),
        "percentages": percentages.tolist(),
        "flags_to_breakouts": flags_to_breakouts,
        "breakouts_to_retests": breakouts_to_retests,
        "retests_to_continuations": retests_to_continuations,
        "min_flag_breakout_gap": _min_gap(flag_breakout_gaps),
        "min_breakout_retest_gap": _min_gap(breakout_retest_gaps),
        "min_retest_continuation_gap": _min_gap(retest_continuation_gaps),
    }

    # Check prevalence against target range
    min_pct = params.min_head_prevalence * 100
    max_pct = params.max_head_prevalence * 100
    summary["out_of_range_heads"] = [
        name for i, name in enumerate(LABEL_NAMES)
        if summary["percentages"][i] < min_pct or summary["percentages"][i] > max_pct
    ]
    return summary


def write_label_report(summary: Dict[str, Any], report_path: Path) -> None:
    lines = [
        f"Total windows: {summary['total']}",
        "Label Distribution:",
    ]
    for name, count, pct in zip(LABEL_NAMES, summary["counts"], summary["percentages"]):
        lines.append(f"- {name}: {int(count)} ({pct:.2f}%)")

    def _fmt(value: Optional[int]) -> str:
        return f"{value} bars" if value is not None else "n/a"

    lines.extend(
        [
            "",
            "Dependencies:",
            f"- Flags leading to breakouts: {summary['flags_to_breakouts']} "
            f"(minimum gap {_fmt(summary['min_flag_breakout_gap'])})",
            f"- Breakouts leading to retests: {summary['breakouts_to_retests']} "
            f"(minimum gap {_fmt(summary['min_breakout_retest_gap'])})",
            f"- Retests leading to continuations: {summary['retests_to_continuations']} "
            f"(minimum gap {_fmt(summary['min_retest_continuation_gap'])})",
        ]
    )
    if summary.get("out_of_range_heads"):
        lines.append("")
        lines.append(
            f"Heads outside 1-5% prevalence range: {', '.join(summary['out_of_range_heads'])} (consider adjusting tolerances)."
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def update_config_paths(config_path: Path, dataset_relative: str) -> None:
    lines = config_path.read_text().splitlines()
    in_data = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_data = stripped == "[data]"
            continue
        if not in_data:
            continue
        if stripped.startswith("train_x"):
            lines[i] = f'train_x = "{dataset_relative}"'
        elif stripped.startswith("train_y"):
            lines[i] = f'train_y = "{dataset_relative}"'
        elif stripped.startswith("val_x"):
            lines[i] = f'val_x = "{dataset_relative}"'
        elif stripped.startswith("val_y"):
            lines[i] = f'val_y = "{dataset_relative}"'
    config_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic rule-based labeling pipeline.")
    parser.add_argument("--dataset", type=str, default="data/dataset.npz", help="Input dataset NPZ.")
    parser.add_argument(
        "--out", type=str, default="data/dataset_labeled.npz", help="Output labeled dataset path."
    )
    parser.add_argument(
        "--preview",
        type=str,
        default="models/cnn_bullflag/plots/label_preview.png",
        help="Preview plot path.",
    )
    parser.add_argument("--num-plots", type=int, default=5, help="Number of sample windows to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling plots.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N windows from the dataset (useful for smoke tests).",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Force enable progress logging while labeling windows.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Force disable progress logging regardless of config.toml.",
    )
    parser.set_defaults(progress=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    preview_path = Path(args.preview)

    arrays = load_dataset(dataset_path)
    X = arrays["X"]
    regimes = arrays.get("regimes")
    print(f"Loaded dataset from {dataset_path} | X shape={X.shape}")

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        limit = min(args.limit, X.shape[0])
        if limit < X.shape[0]:
            print(f"Applying --limit={args.limit}; truncated to first {limit} windows.")
        X = X[:limit]
        if regimes is not None:
            regimes = regimes[:limit]

    params = load_labeler_params()
    if args.progress is None:
        show_progress = load_system_show_progress()
    else:
        show_progress = args.progress
    labels, diagnostics = label_dataset(
        X,
        params=params,
        return_diagnostics=True,
        show_progress=show_progress,
    )
    assert diagnostics is not None
    assert labels.shape[0] == X.shape[0]
    assert labels.shape[1] == len(LABEL_NAMES)
    if np.isnan(labels).any():
        raise ValueError("NaNs detected in labels.")

    save_kwargs = {"X": X, "y": labels}
    if regimes is not None:
        save_kwargs["regimes"] = regimes
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **save_kwargs)
    print(f"Saved labeled dataset to {out_path}")

    summary = summarize_labels(labels, diagnostics, params)
    for name, count, pct in zip(LABEL_NAMES, summary["counts"], summary["percentages"]):
        print(f"{name}: {int(count)} positives ({pct:.2f}%)")
    if summary["out_of_range_heads"]:
        print("Heads outside 1-5% prevalence range:", ", ".join(summary["out_of_range_heads"]))
        print("Consider adjusting tolerances in [labeler] config.")
    report_path = Path("models/cnn_bullflag/label_report.txt")
    write_label_report(summary, report_path)
    print(f"Wrote label distribution report to {report_path}")

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.num_plots, X.shape[0])
    sample_indices = rng.choice(X.shape[0], size=sample_count, replace=False)
    _plot_sample_windows(X, labels, sample_indices, preview_path)
    print(f"Saved preview plot to {preview_path}")

    update_config_paths(Path("config.toml"), dataset_relative=str(out_path).replace("\\", "/"))
    print("Updated config.toml [data] paths.")


if __name__ == "__main__":
    main()
