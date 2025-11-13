"""CPU tuning helper for PyTorch thread count on Intel CPUs.

Runs quick forward passes through the stub model with various
torch.set_num_threads values and reports throughput.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cnn_bullflag.model_stub import BullFlagCNNMultiHeadStub

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def run_once(model: torch.nn.Module, batch: torch.Tensor, iters: int = 100) -> float:
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch)
    start = time.perf_counter()
    with torch.no_grad():
        rng = range(iters)
        if tqdm is not None:
            rng = tqdm(rng, leave=False, desc="Benchmark", total=iters)
        for _ in rng:  # type: ignore
            _ = model(batch)
    dur = time.perf_counter() - start
    items = batch.size(0) * iters
    return items / dur


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BullFlagCNNMultiHeadStub().to(device).eval()
    batch = torch.from_numpy(np.random.normal(size=(32, 12, 200)).astype(np.float32)).to(device)

    candidates = [1, 2, 4, 6, 8, 12]
    results = []
    for t in candidates:
        try:
            torch.set_num_threads(t)
        except Exception:
            continue
        thpt = run_once(model, batch, iters=100)
        results.append((t, thpt))
        print(f"threads={t:>2} -> {thpt:8.1f} items/sec")

    if results:
        best_threads, best = max(results, key=lambda x: x[1])
        print(f"Recommended torch.set_num_threads({best_threads}) for this workload.")


if __name__ == "__main__":
    main()

