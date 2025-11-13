"""Run Grad-CAM on a random window to ensure hooks and shapes work."""

import numpy as np
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._setup_path import add_project_root

add_project_root()

from models.cnn_bullflag.infer import BullFlagCNNInfer
from models.cnn_bullflag.model_stub import BullFlagCNNMultiHeadStub


def main() -> None:
    rng = np.random.default_rng(123)
    feature_window = rng.normal(size=(12, 200)).astype(np.float32)
    model = BullFlagCNNMultiHeadStub()
    infer = BullFlagCNNInfer(model=model, enable_gradcam=True)
    result = infer(feature_window)
    activation_map = result["activation_map"]
    assert activation_map is not None, "Grad-CAM should be enabled"
    min_val = min(activation_map)
    max_val = max(activation_map)
    print(f"Sequence score: {result['sequence_score']:.4f}")
    print(f"Activation map length: {len(activation_map)}")
    print(f"Activation min/max: {min_val:.4f}/{max_val:.4f}")
    if min_val < -1e-5 or max_val > 1 + 1e-5:
        raise AssertionError("Activation map not in [0,1]")


if __name__ == "__main__":
    main()
