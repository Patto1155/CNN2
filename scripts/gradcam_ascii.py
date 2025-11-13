"""ASCII visualization of Grad-CAM activation across a window."""

import numpy as np
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._setup_path import add_project_root

add_project_root()

from models.cnn_bullflag.infer import BullFlagCNNInfer
from models.cnn_bullflag.model_stub import BullFlagCNNMultiHeadStub


def render_ascii(activation: list[float], width: int = 80) -> str:
    arr = np.array(activation)
    arr = (arr - arr.min()) / max(float(np.ptp(arr)), 1e-8)
    bins = np.linspace(0, arr.size, width + 1, dtype=int)
    chars = " .:-=+*#%@"
    segments = []
    for i in range(width):
        segment = arr[bins[i] : bins[i + 1]]
        value = segment.mean() if segment.size else 0.0
        idx = min(int(value * (len(chars) - 1)), len(chars) - 1)
        segments.append(chars[idx])
    return "".join(segments)


def main() -> None:
    rng = np.random.default_rng(2024)
    feature_window = rng.normal(size=(12, 200)).astype(np.float32)
    infer = BullFlagCNNInfer(model=BullFlagCNNMultiHeadStub(), enable_gradcam=True)
    result = infer(feature_window)
    print(f"Sequence score: {result['sequence_score']:.3f}")
    print("Grad-CAM ASCII heat:")
    print(render_ascii(result["activation_map"]))


if __name__ == "__main__":
    main()
