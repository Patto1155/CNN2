"""
Config-driven training scaffold for the BullFlag multi-head CNN.

Reads dataset paths and hyperparameters from config.toml and performs a
minimal training loop using the model stub. Saves a checkpoint and
validation outputs (y_true/y_probs) for downstream metrics reporting.

Dataset format (canonical): NPZ with keys
  - X: float32 array of shape [N, C, L]
  - y: int8/float32 array of shape [N, 4] (binary targets)
  - regimes (optional): array of shape [N] with strings

Also supports separate .npy files when config points to explicit paths
for X and y (e.g., train_x.npy, train_y.npy).

Run:
  python scripts/train_real.py

All behavior is driven from config.toml — no hard-coded paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts._setup_path import add_project_root


add_project_root()

from scripts.config import load_config  # noqa: E402
from models.cnn_bullflag.model import build_model_from_config  # noqa: E402

try:  # tqdm is optional
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass
class Normalizer:
    mean: np.ndarray  # [C, 1]
    std: np.ndarray   # [C, 1]

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.clip(self.std, 1e-8, None)


def _load_xy(x_path: Path, y_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    def load_any(p: Path) -> np.ndarray:
        if p.suffix == ".npz":
            data = np.load(p, allow_pickle=False)
            if "X" in data:
                return data["X"]
            if "arr_0" in data:
                return data["arr_0"]
            raise ValueError(f"NPZ file {p} must contain 'X' or 'arr_0'")
        if p.suffix == ".npy":
            return np.load(p, allow_pickle=False)
        raise ValueError(f"Unsupported file: {p}")

    def load_y(p: Path) -> np.ndarray:
        if p.suffix == ".npz":
            data = np.load(p, allow_pickle=False)
            if "y" in data:
                return data["y"]
            if "arr_0" in data:
                return data["arr_0"]
            raise ValueError(f"NPZ file {p} must contain 'y' or 'arr_0'")
        if p.suffix == ".npy":
            return np.load(p, allow_pickle=False)
        raise ValueError(f"Unsupported file: {p}")

    X = load_any(x_path)
    y = load_y(y_path)
    if X.ndim != 3:
        raise ValueError(f"X must be [N,C,L]; got {X.shape}")
    if y.shape[-1] != 4:
        raise ValueError(f"y must have last dim 4; got {y.shape}")
    return X.astype(np.float32), y.astype(np.float32)


def _compute_normalizer(X_train: np.ndarray) -> Normalizer:
    # Compute per-channel mean/std across N and L
    C = X_train.shape[1]
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # [1, C, 1]
    std = X_train.std(axis=(0, 2), keepdims=True)    # [1, C, 1]
    return Normalizer(mean=mean.reshape(C, 1), std=std.reshape(C, 1))


def _make_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    x_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def _bce_multihead(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # logits shape: [B, 4]; y shape: [B, 4]
    return nn.functional.binary_cross_entropy_with_logits(logits, y.float())


def main() -> None:
    cfg = load_config()

    threads = int(cfg.get("system", "threads", default=0) or 0)
    if threads > 0:
        torch.set_num_threads(threads)

    show_progress: bool = bool(cfg.get("system", "show_progress", default=True))

    train_x = Path(str(cfg.get("data", "train_x", default="")))
    train_y = Path(str(cfg.get("data", "train_y", default="")))
    val_x = Path(str(cfg.get("data", "val_x", default="")))
    val_y = Path(str(cfg.get("data", "val_y", default="")))
    if not (train_x.exists() and train_y.exists() and val_x.exists() and val_y.exists()):
        raise SystemExit("Please set [data].train_x/train_y/val_x/val_y in config.toml")

    Xtr, ytr = _load_xy(train_x, train_y)
    Xva, yva = _load_xy(val_x, val_y)

    norm = _compute_normalizer(Xtr)
    Xtr_n = norm.apply(Xtr)
    Xva_n = norm.apply(Xva)

    batch_size = int(cfg.get("train", "batch_size", default=32) or 32)
    epochs = int(cfg.get("train", "epochs", default=0) or 0)
    steps = int(cfg.get("train", "steps", default=0) or 0)
    lr = float(cfg.get("train", "learning_rate", default=1e-3) or 1e-3)

    train_loader = _make_loader(Xtr_n, ytr, batch=batch_size, shuffle=True)
    val_loader = _make_loader(Xva_n, yva, batch=batch_size, shuffle=False)

    device = torch.device("cpu")
    window_len = Xtr.shape[2]
    in_channels = Xtr.shape[1]
    model = build_model_from_config(window_len=window_len, in_channels=in_channels, num_heads=4)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    log = {"epochs": epochs or None, "steps": steps or None, "batch_size": batch_size, "lr": lr}

    # Placeholder training loop: iterate epochs if provided; otherwise run for N steps
    def _iter_loader():
        iterator = train_loader
        if show_progress and tqdm is not None:
            iterator = tqdm(train_loader, desc="Train", unit="batch")  # type: ignore
        for xb, yb in iterator:  # type: ignore
            yield xb.to(device), yb.to(device)

    total_steps = 0
    if epochs > 0:
        for _ in range(epochs):
            for xb, yb in _iter_loader():
                opt.zero_grad(set_to_none=True)
                logits = model.forward_logits(xb)
                loss = _bce_multihead(logits, yb)
                loss.backward()
                opt.step()
                total_steps += 1
    else:
        target_steps = steps if steps > 0 else 100
        while total_steps < target_steps:
            for xb, yb in _iter_loader():
                opt.zero_grad(set_to_none=True)
                logits = model.forward_logits(xb)
                loss = _bce_multihead(logits, yb)
                loss.backward()
                opt.step()
                total_steps += 1
                if total_steps >= target_steps:
                    break

    # Simple validation forward to collect y_true and y_probs
    model.eval()
    all_probs = []
    all_true = []
    with torch.no_grad():
        iterator = val_loader
        if show_progress and tqdm is not None:
            iterator = tqdm(val_loader, desc="Val", unit="batch")  # type: ignore
        for xb, yb in iterator:  # type: ignore
            out = model(xb.to(device))
            probs = torch.stack(
                [
                    out["flag"].view(-1),
                    out["breakout"].view(-1),
                    out["retest"].view(-1),
                    out["continuation"].view(-1),
                ],
                dim=1,
            )
            all_probs.append(probs.cpu().numpy())
            all_true.append(yb.numpy())
    y_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 4), dtype=np.float32)
    y_true = np.concatenate(all_true, axis=0) if all_true else np.zeros((0, 4), dtype=np.float32)

    # Save artifacts
    ckpt_path = Path(str(cfg.get("checkpoint", "model_path", default="models/cnn_bullflag/checkpoint.pt")))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_channels": Xtr.shape[1]}, ckpt_path)

    eval_dir = Path(str(cfg.get("paths", "eval_dir", default="models/cnn_bullflag")))
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.save(eval_dir / "y_true.npy", y_true)
    np.save(eval_dir / "y_probs.npy", y_probs)

    # Training log
    log.update({"steps_run": total_steps, "train_size": int(Xtr.shape[0]), "val_size": int(Xva.shape[0])})
    (eval_dir / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved eval arrays to {eval_dir}")


if __name__ == "__main__":
    main()
