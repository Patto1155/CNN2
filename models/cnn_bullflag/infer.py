"""Inference helper that optionally runs Grad-CAM and builds schema outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np
import torch

from .gradcam import GradCAM1D
from .model import BullFlagCNNMultiHead
from .model_stub import BullFlagCNNMultiHeadStub
from .schema import build_cnn_output
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _geometric_mean(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vals = values + eps
    prod = torch.prod(vals)
    return prod.pow(1.0 / float(vals.numel()))


@dataclass
class WindowMeta:
    window_id: str
    symbol: str
    timeframe: str
    start_ts: int
    end_ts: int


class BullFlagCNNInfer:
    """Runs the CNN, computes sequence score, and optional Grad-CAM."""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: str = "cpu",
        enable_gradcam: bool = False,
        gradcam_target_layer: str = "conv4",
    ) -> None:
        self.device = device
        self.model = model or BullFlagCNNMultiHeadStub()
        self.model.to(self.device)
        self.model.eval()

        self.enable_gradcam = enable_gradcam
        self.gradcam: Optional[GradCAM1D] = None
        if enable_gradcam:
            self.gradcam = GradCAM1D(
                self.model, target_layer_name=gradcam_target_layer, device=self.device
            )

    def _to_tensor(self, feature_window: np.ndarray) -> torch.Tensor:
        if feature_window.ndim != 2:
            raise ValueError(
                f"feature_window must be [n_features, seq_len]; got {feature_window.shape}"
            )
        tensor = torch.from_numpy(feature_window).float().unsqueeze(0)
        return tensor.to(self.device)

    def __call__(self, feature_window: np.ndarray) -> Dict[str, Any]:
        x = self._to_tensor(feature_window)
        with torch.no_grad():
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                _, probs = outputs  # real model returns (logits, probs)
            else:
                probs = outputs  # stub model returns dict

        if isinstance(probs, dict):
            # Stub model
            def _extract(name: str) -> float:
                tensor = probs[name]
                if tensor.ndim == 0:
                    value = tensor
                else:
                    value = tensor.view(-1)[0]
                return float(value.detach().cpu())

            flag = _extract("flag")
            breakout = _extract("breakout")
            retest = _extract("retest")
            continuation = _extract("continuation")
        else:
            # Real model: probs is [1, 4]
            probs_flat = probs.view(-1).cpu()
            flag = float(probs_flat[0])
            breakout = float(probs_flat[1])
            retest = float(probs_flat[2])
            continuation = float(probs_flat[3])

        probs_tensor = torch.tensor([flag, breakout, retest, continuation], device=self.device)
        sequence_score = float(_geometric_mean(probs_tensor).cpu())

        activation_map = None
        if self.enable_gradcam and self.gradcam is not None:
            x_grad = x.detach().clone().requires_grad_(True)
            cam = self.gradcam.generate_cam(x_grad, target_head="sequence")
            activation_map = cam.numpy().tolist()

        return {
            "flag_prob": flag,
            "breakout_prob": breakout,
            "retest_prob": retest,
            "continuation_prob": continuation,
            "sequence_score": sequence_score,
            "activation_map": activation_map,
        }

    def infer_with_schema(self, feature_window: np.ndarray, meta: WindowMeta) -> Dict[str, Any]:
        infer_result = self(feature_window)
        return build_cnn_output(
            window_id=meta.window_id,
            symbol=meta.symbol,
            tf=meta.timeframe,
            start_ts=meta.start_ts,
            end_ts=meta.end_ts,
            infer_result=infer_result,
        )


def run_inference_batch(
    feature_windows: np.ndarray,
    metas: Optional[list[WindowMeta]] = None,
    enable_gradcam: bool = False,
    output_path: Optional[Path] = None,
    show_progress: bool = True,
) -> list[Dict[str, Any]]:
    """Helper to run inference over a batch and optionally persist JSON lines."""

    if feature_windows.ndim != 3:
        raise ValueError(f"feature_windows must be [N, C, L]; got {feature_windows.shape}")

    num_windows = feature_windows.shape[0]
    if metas is None:
        metas = [
            WindowMeta(
                window_id=f"window_{idx:04d}",
                symbol="DEMO",
                timeframe="5m",
                start_ts=idx * 60,
                end_ts=(idx + 1) * 60,
            )
            for idx in range(num_windows)
        ]
    if len(metas) != num_windows:
        raise ValueError("Number of metas must match number of windows")

    infer = BullFlagCNNInfer(enable_gradcam=enable_gradcam)
    outputs = []
    iterator = zip(feature_windows, metas)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, total=num_windows, desc="Inference", unit="win")  # type: ignore
    for window, meta in iterator:  # type: ignore
        outputs.append(infer.infer_with_schema(window, meta))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for row in outputs:
                fh.write(json.dumps(row))
                fh.write("\n")

    return outputs
