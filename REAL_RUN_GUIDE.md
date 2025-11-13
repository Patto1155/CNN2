Real Run Guide â€” TradingCNN

This guide explains how to run the system endâ€‘toâ€‘end with your own model/data, plus three sample training configurations (CPU light, CPU tuned, GPU).

Prereqs

- Python 3.9+
- pip installs: `pip install -r requirements.txt` then PyTorch for your platform
  - CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` (or see pytorch.org)

1) Integrate Your Model

- Replace the stub with your CNN (multiâ€‘head outputs):
  - Expected forward return: {"flag": T, "breakout": T, "retest": T, "continuation": T} where each T is scalar or [B,1].
- Ensure the last conv layer you want to explain is named (e.g., conv4) or pass its name to GradCAM1D.
- Wire inference with Gradâ€‘CAM using BullFlagCNNInfer:
  - infer = BullFlagCNNInfer(model=YourModel(), enable_gradcam=True, gradcam_target_layer="conv4")

2) Prepare Data

- Training data:
  - Provide a DataLoader that yields tensors [B, C, L] and binary targets [B, 4] for the four heads.
  - Normalize features consistently between train and inference.

- Evaluation data:
  - For metrics: produce y_true: [N, 4] (ints) and y_probs: [N, 4] (floats 0..1) on a heldâ€‘out set.
  - Optionally produce regimes: [N] strings like trending_up | trending_down | choppy.

3) Train

- Use your own training loop or adapt scripts/full_cycle_mock.py as a reference (loss is average of BCE across heads).
- Save your model checkpoint.

4) Evaluate & Report

- Compute/save metrics JSON:
  - python -m models.cnn_bullflag.eval_metrics --ytrue path/to/y_true.npy --yprobs path/to/y_probs.npy --regimes path/to/regimes.npy
- Print tables:
  - python -m models.cnn_bullflag.report_metrics

5) Inference at Scale

- Batch helper (add Gradâ€‘CAM if needed):
  - Use run_inference_batch(feature_windows, metas, enable_gradcam=True, output_path=...)
- Output rows follow the standard schema (see README).

6) Visualize

- ASCII preview: python scripts/gradcam_ascii.py
- PNG overlay (with optional price CSV):
  - python scripts/plot_gradcam_overlay.py --jsonl models/cnn_bullflag/example_cnn_outputs.jsonl --index 0 --out models/cnn_bullflag/plots/window_0000.png
  - Add your price series: --price-csv path/to/prices.csv --price-col close

Sample Training Configurations

Config A â€” CPU Light (Laptopâ€‘friendly)
- Hardware: Intel 12thâ€‘gen i5 (10C/12T), no GPU
- Threads: torch.set_num_threads(6â€“8) (scripts default to <=8 on CPU)
- Hyperparams (example):
  - Batch size: 32
  - Steps: 10k (or epochs ~10 depending on dataset)
  - LR: 1eâ€‘3, Optimizer: Adam
  - Gradâ€‘CAM: disabled during full sweeps; enable only for sampled windows
- Commands:
  - Train: your training script (adapt from scripts/full_cycle_mock.py)
  - Eval metrics: as in section 4
  - Inference dump: python scripts/generate_cnn_output_demo.py (replace with your pipeline)

Config B â€” CPU Tuned (Heavier Runs)
- Hardware: Multiâ€‘core CPU
- Tune threads using: python scripts/cpu_tuning.py then set OMP/MKL vars
  - Windows PowerShell: setx OMP_NUM_THREADS 8, setx MKL_NUM_THREADS 8
- Hyperparams:
  - Batch size: 64 (if memory allows)
  - Steps/Epochs: 50% more than Config A
  - Learning rate schedule: Cosine or step decay
  - Enable mixed precision if supported by your CPU BLAS (optional)

Config C â€” GPU (Fast)
- Hardware: NVIDIA GPU with CUDA 12.x
- Install CUDA PyTorch wheel and enable mixed precision
- Hyperparams:
  - Batch size: 128â€“256
  - Epochs: 20â€“50 (datasetâ€‘dependent)
  - AMP: use torch.cuda.amp.autocast() and GradScaler
  - Gradâ€‘CAM: compute for topâ€‘K windows per epoch to inspect learning focus
- Skeleton:
scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        out = model(x)
        loss = bce_heads(out, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

Operational Notes

- Thresholds: metrics at fixed thresholds can be brittle; tune per head on validation.
- Calibration: add reliability plots if you need wellâ€‘calibrated scores.
- Performance: Gradâ€‘CAM does an extra backward pass; gate it to keep throughput high.
- Reproducibility: seed RNGs and log dataset/model versions in the meta field.


Real Pipelines (Config)

- Train: `python scripts/train_real.py`
- Inference: `python scripts/infer_real.py`
- Metrics: `python -m models.cnn_bullflag.eval_metrics --ytrue models/cnn_bullflag/y_true.npy --yprobs models/cnn_bullflag/y_probs.npy --output-dir models/cnn_bullflag`
- Gallery: `python scripts/plot_gallery.py --jsonl models/cnn_bullflag/example_cnn_outputs.jsonl --out-dir models/cnn_bullflag/plots/gallery --top-n 12 --sort-by sequence_score`

Config Profiles

CPU-light
- `[system].threads = 4`
- `[train]` small: `batch_size=16`, `epochs=0`, `steps=200`, `learning_rate=0.001`
- Run:
  - `python scripts/train_real.py`
  - `python -m models.cnn_bullflag.eval_metrics --ytrue models/cnn_bullflag/y_true.npy --yprobs models/cnn_bullflag/y_probs.npy --output-dir models/cnn_bullflag`
  - `python scripts/infer_real.py`
  - `python scripts/plot_gallery.py --jsonl models/cnn_bullflag/example_cnn_outputs.jsonl --out-dir models/cnn_bullflag/plots/gallery --top-n 12 --sort-by sequence_score`

CPU-balanced
- Tune with `python scripts/cpu_tuning.py` and set `[system].threads` accordingly (e.g., 6-8)
- `[train]` medium: `batch_size=32`, `epochs=2`, `learning_rate=0.001`
- Run same commands as above.

Full-training
- `[system].threads = 0` to let PyTorch decide or set to your tuned value
- `[train]` larger: `batch_size=64`, `epochs=10+`, `learning_rate=0.001` (or schedule)
- For inference with CSV windows, set `[inference].price_csv`, `window_len`, `stride` and leave `input_dir` empty.
- Run same command sequence.

Dataset Format
- Canonical NPZ with keys: `X:[N,C,L]`, `y:[N,4]`, `regimes` (optional)
- Alternatively point `train_x`, `train_y`, `val_x`, `val_y` to .npy/.npz files directly.

Notes
- All paths, hyperparameters, and outputs are driven by `config.toml`.
- The current code uses a model stub; swap in your real CNN and loss when ready.
