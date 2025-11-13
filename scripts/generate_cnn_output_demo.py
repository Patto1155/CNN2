"""Generate demo CNN outputs with schema-wrapped inference results."""

import json
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._setup_path import add_project_root

add_project_root()

from models.cnn_bullflag.infer import run_inference_batch

OUTPUT_PATH = Path("models/cnn_bullflag/example_cnn_outputs.jsonl")


def main(num_windows: int = 5) -> None:
    rng = np.random.default_rng(99)
    feature_windows = rng.normal(size=(num_windows, 12, 200)).astype(np.float32)
    outputs = run_inference_batch(
        feature_windows,
        enable_gradcam=True,
        output_path=OUTPUT_PATH,
        show_progress=True,
    )
    print(f"Wrote {len(outputs)} windows to {OUTPUT_PATH}")
    print("First record:")
    print(json.dumps(outputs[0], indent=2))


if __name__ == "__main__":
    main()
