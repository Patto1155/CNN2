TradingCNN â€” BullFlag CNN Metrics + Gradâ€‘CAM

This repo contains a lightweight, endâ€‘toâ€‘end scaffold for evaluating a multiâ€‘head 1D CNN on trading windows, computing perâ€‘head metrics, and generating Gradâ€‘CAM activation maps for visual inspection.

It includes:
- Perâ€‘head metrics with optional perâ€‘regime breakdown
- 1D Gradâ€‘CAM integration targeting a named conv layer
- A standard JSON output schema for downstream apps
- Smoke tests, reporting CLIs, and demo scripts

If youâ€™re not familiar with PyTorch or Gradâ€‘CAM, the quickstart below walks you through running everything with a selfâ€‘contained model stub.

Quickstart (Bash)

- Create and activate a virtual environment
  - macOS/Linux
    - `python3 -m venv .venv && source .venv/bin/activate`
  - Windows (PowerShell)
    - `python -m venv .venv; .venv\Scripts\Activate.ps1`

- Upgrade pip
  - `python -m pip install --upgrade pip`

- Install dependencies
  - Install NumPy and helpers from `requirements.txt`:
    - `pip install -r requirements.txt`
  - Install PyTorch (choose one):
    - CPU only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
    - CUDA 12.x: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
    - For other platforms: see https://pytorch.org/get-started/locally/

CPUâ€‘only (Intel 12thâ€‘gen) tips:
- Your i5â€‘1235U (10C/12T) is fine for demos and light training.
- Set PyTorch threads to a sensible value for CPU runs:
  - In scripts, we default to up to 8 threads; you can tune via:
    - `python scripts/cpu_tuning.py` (prints recommended torch.set_num_threads)
  - You can also set env vars for BLAS libraries if needed:
    - `set OMP_NUM_THREADS=8` (Windows) / `export OMP_NUM_THREADS=8` (bash)
    - `set MKL_NUM_THREADS=8` (Windows) / `export MKL_NUM_THREADS=8` (bash)

- Run synthetic metrics (produces JSON metrics files)
  - `python scripts/test_metrics_synthetic.py`

- Print metrics tables from saved JSON
  - `python -m models.cnn_bullflag.report_metrics`

- Smokeâ€‘test Gradâ€‘CAM on a random window
  - `python scripts/smoke_gradcam.py`

- Generate demo JSONL outputs with activation maps and scores
  - `python scripts/generate_cnn_output_demo.py`

- Summarize top windows and activation hotspots
  - `python scripts/analytics_demo.py`

- ASCII preview of activation map heat
  - `python scripts/gradcam_ascii.py`

- Simulate a full mock cycle (toy training, evaluation + metrics, Gradâ€‘CAM)
  - `python scripts/full_cycle_mock.py` (add `--no-progress` to disable bars)
  - Uses `config.toml` for defaults (e.g., threads, steps); CLI flags override

- Optional: demo data download with progress bar
  - `python scripts/download_demo_data.py`
  
- Optional: CPU tuning helper (finds good thread count)
  - `python scripts/cpu_tuning.py`
  - Update `config.toml` `[system].threads` with the recommended value (or run: `python scripts/full_cycle_mock.py --threads <n>`) 

If any script complains about imports, run it from the project root (where this README.md lives). You can also set `export PYTHONPATH=.` before running scripts.

Repository Map

- `utils/metrics.py` â€” metrics for single head, multiâ€‘head, perâ€‘regime
- `models/cnn_bullflag/gradcam.py` â€” 1D Gradâ€‘CAM utility with hooks
- `models/cnn_bullflag/model_stub.py` â€” minimal CNN with a named `conv4` layer (for demos)
- `models/cnn_bullflag/infer.py` â€” inference wrapper, optional Gradâ€‘CAM, schema output
  - Includes a batch helper with a tqdm progress bar
- `models/cnn_bullflag/schema.py` â€” helper to build standard JSON payloads
- `models/cnn_bullflag/eval_metrics.py` â€” compute + save metrics JSON
- `models/cnn_bullflag/report_metrics.py` â€” CLI to print metrics tables
- `scripts/` â€” smoke tests and demos (metrics, Gradâ€‘CAM, analytics, full cycle)
  - Also contains download helpers with progress (`scripts/download_utils.py`)
- `sequence_score_decision.md` â€” rationale for sequence score aggregation

What The Outputs Look Like

`run_inference_batch` and `generate_cnn_output_demo.py` write JSON Lines to `models/cnn_bullflag/example_cnn_outputs.jsonl` with the following structure per window:

- `window_id` / `symbol` / `tf` / `window_start_ts` / `window_end_ts`
- `scores`:
  - `flag_prob`, `breakout_prob`, `retest_prob`, `continuation_prob`
  - `sequence_score` (geometric mean across heads)
- `activation_map` (optional):
  - `indices` (0..seq_lenâ€‘1)
  - `intensities` (values in [0, 1])
- `meta`:
  - `model_version`

Metrics are written to:
- `models/cnn_bullflag/test_multihead_metrics.json`
- `models/cnn_bullflag/test_multihead_metrics_by_regime.json`

Use the reporting CLI to print readable tables:
- `python -m models.cnn_bullflag.report_metrics`

Integrating Your Own Model

Replace the stub with your real multiâ€‘head CNN.

- Ensure your model returns a dict with keys: `flag`, `breakout`, `retest`, `continuation` (each a scalar tensor or `[B,1]`).
- Name the last convolutional layer you want to explain as `conv4` (or pass a custom name into GradCAM1D).
- Example wiring:

```
from models.cnn_bullflag.infer import BullFlagCNNInfer, WindowMeta

model = YourBullFlagCNN()
infer = BullFlagCNNInfer(model=model, enable_gradcam=True, gradcam_target_layer="conv4")

# Single window [C, L]
result = infer(feature_window_np)

# With schema wrapping
meta = WindowMeta(window_id="abc", symbol="ES", timeframe="5m", start_ts=..., end_ts=...)
row = infer.infer_with_schema(feature_window_np, meta)
```

If your target layer has a different name, pass it to the class:
- `GradCAM1D(model, target_layer_name="your_layer_name", device=...)`

Tips & Troubleshooting

- Import errors when running scripts:
  - Run from repo root and/or set `export PYTHONPATH=.`
- Gradâ€‘CAM returns a flat/nearâ€‘zero map:
  - This is expected for random weights; try after training or use the mock trainer (`scripts/full_cycle_mock.py`).
- Metrics look odd at fixed thresholds:
  - Use the `scripts/inspect_metrics.py` helper and consider tuning thresholds per head based on validation.
- Performance:
  - Gradâ€‘CAM does an extra backward pass; gate it for topâ€‘K windows or only for debugging.

Requirements

- Python 3.9+
- NumPy (2.0+ supported)
- PyTorch 2.x (CPU or CUDA build)
- Matplotlib only if you add your own visual plots (not required by included scripts)

Install core packages with:
- `pip install -r requirements.txt`
- Then install PyTorch with the appropriate index URL as shown in Quickstart.

Next Steps

- Swap in your trained model and data loader to replace the stub network.
- Store schema rows into your datastore (Parquet/SQLite/etc.) and overlay `activation_map` on charts.
- Add ROCâ€‘AUC/PRâ€‘AUC to metrics if needed; extend reporting CLI to include calibration plots.

Real Training + Inference (Config-driven)

- Configure paths and hyperparameters in `config.toml`:
  - `[data]`: `train_x`, `train_y`, `val_x`, `val_y` (NPZ/NPY)
  - `[train]`: `epochs` or `steps`, `batch_size`, `learning_rate`
  - `[checkpoint]`: `model_path`
  - `[inference]`: set either `input_dir` (windows) or `price_csv` (CSV mode), plus `output_jsonl`
  - `[paths]`: `eval_dir` (where y_true/y_probs + training_log.json are saved)

- Canonical dataset format (NPZ):
  - `X`: float32 of shape `[N, C, L]`
  - `y`: int/bool/float of shape `[N, 4]` (binary labels per head)
  - `regimes` (optional): shape `[N]` strings

- Train:
  - `python scripts/train_real.py`

- Inference (JSONL with optional Grad-CAM):
  - `python scripts/infer_real.py`

- Evaluate metrics from saved arrays (compatible with report CLI):
  - `python -m models.cnn_bullflag.eval_metrics --ytrue models/cnn_bullflag/y_true.npy --yprobs models/cnn_bullflag/y_probs.npy --output-dir models/cnn_bullflag`
  - Print tables: `python -m models.cnn_bullflag.report_metrics`

- Build a Grad-CAM gallery from inference output:
  - `python scripts/plot_gallery.py --jsonl models/cnn_bullflag/example_cnn_outputs.jsonl --out-dir models/cnn_bullflag/plots/gallery --top-n 12 --sort-by sequence_score`
