"""Simulated end-to-end cycle: train stub, evaluate, run Grad-CAM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import os

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._setup_path import add_project_root

add_project_root()

from models.cnn_bullflag.eval_metrics import save_metrics_to_json
from models.cnn_bullflag.infer import BullFlagCNNInfer, WindowMeta
from models.cnn_bullflag.model_stub import BullFlagCNNMultiHeadStub
from scripts.config import load_config
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def train_stub(
    model: BullFlagCNNMultiHeadStub,
    device: str = "cpu",
    steps: int = 50,
    show_progress: bool = True,
) -> None:
    rng = torch.Generator().manual_seed(0)
    n = 256
    features = torch.randn((n, 12, 200), generator=rng, device=device)
    labels = torch.randint(0, 2, (n, 4), generator=rng, device=device).float()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    step = 0
    while step < steps:
        iterator = loader
        if show_progress and tqdm is not None:
            iterator = tqdm(loader, total=len(loader), desc=f"Train epoch", leave=False)
        for batch_x, batch_y in iterator:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            losses = []
            for idx, head in enumerate(["flag", "breakout", "retest", "continuation"]):
                target = batch_y[:, idx].unsqueeze(1)
                losses.append(F.binary_cross_entropy(outputs[head], target))
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()
            step += 1
            if not (show_progress and tqdm is not None) and step % 10 == 0:
                print(f"Train step {step}: loss={loss.item():.4f}")
            if step >= steps:
                break


def evaluate_stub(
    model: BullFlagCNNMultiHeadStub,
    device: str = "cpu",
    show_progress: bool = True,
) -> None:
    model.eval()
    rng = np.random.default_rng(123)
    n = 128
    features = rng.normal(size=(n, 12, 200)).astype(np.float32)
    y_true = rng.integers(0, 2, size=(n, 4))
    regimes = rng.choice(["trending_up", "trending_down", "choppy"], size=n)

    infer = BullFlagCNNInfer(model=model, device=device, enable_gradcam=True)
    probs = []
    iterator = enumerate(features)
    if show_progress and tqdm is not None:
        iterator = enumerate(tqdm(features, total=n, desc="Eval inference", unit="win"))
    for idx, window in iterator:
        result = infer(window)
        probs.append(
            [
                result["flag_prob"],
                result["breakout_prob"],
                result["retest_prob"],
                result["continuation_prob"],
            ]
        )
        if idx < 3 and result["activation_map"]:
            print(
                f"Window {idx} activation sample: {result['activation_map'][:5]} ..."
            )
    y_probs = np.array(probs)
    save_metrics_to_json(y_true=y_true, y_probs=y_probs, regimes=regimes)
    print("Saved metrics JSON via evaluate_stub")

    metas = [
        WindowMeta(
            window_id=f"eval_{idx:03d}",
            symbol="DEMO",
            timeframe="5m",
            start_ts=idx * 60,
            end_ts=(idx + 1) * 60,
        )
        for idx in range(min(5, n))
    ]
    _ = infer.infer_with_schema(features[0], metas[0])


def main() -> None:
    import argparse

    cfg = load_config()

    parser = argparse.ArgumentParser(description="Full mock cycle with progress bars")
    parser.add_argument("--steps", type=int, default=None, help="Training steps (override config)")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bars"
    )
    parser.add_argument("--threads", type=int, help="Override torch.set_num_threads on CPU")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reasonable default threads for CPU; allow explicit override via --threads
    if device == "cpu":
        try:
            threads_cfg = int(cfg.get("system", "threads", default=0) or 0)
            if args.threads and args.threads > 0:
                torch.set_num_threads(args.threads)
            else:
                if threads_cfg > 0:
                    torch.set_num_threads(threads_cfg)
                else:
                    logical = os.cpu_count() or 8
                    torch.set_num_threads(min(logical, 8))
        except Exception:
            pass
    model = BullFlagCNNMultiHeadStub().to(device)
    show_cfg = bool(cfg.get("system", "show_progress", default=True))
    show = show_cfg and (not args.no_progress)
    steps = int(args.steps) if args.steps is not None else int(cfg.get("train", "steps", default=50))
    train_stub(model, device=device, steps=steps, show_progress=show)
    evaluate_stub(model, device=device, show_progress=show)
    print("Full mock cycle complete. Use report_metrics CLI to inspect outputs.")


if __name__ == "__main__":
    main()
