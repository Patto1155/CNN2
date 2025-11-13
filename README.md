# TradingCNN — Bull Flag Pattern Detection with CNN

A complete end-to-end system for detecting bull flag patterns in cryptocurrency trading data using a multi-head 1D CNN. Includes per-head evaluation metrics, Grad-CAM visualization, data labeling pipeline, and a web dashboard for monitoring training progress.

## Objective

This project implements a CNN-based system to detect sequential bull flag patterns in BTCUSDT 30-minute price data. The model predicts four sequential stages:
1. **Flag** — Consolidation phase after an impulse move
2. **Breakout** — Price breaks above flag resistance
3. **Retest** — Pullback to breakout level with reclaim
4. **Continuation** — Further upward movement after retest

The system provides:
- **Per-head metrics** to evaluate performance on each pattern stage
- **Grad-CAM heatmaps** to visualize which time steps the model focuses on
- **Rule-based labeling** pipeline for generating training data
- **Web dashboard** for real-time training monitoring and visualization

## Quick Start

### Installation

1. **Create and activate virtual environment:**
   ```bash
   # macOS/Linux
   python3 -m venv .venv && source .venv/bin/activate
   
   # Windows (PowerShell)
   python -m venv .venv; .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   
   # Install PyTorch (choose one):
   # CPU only
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # CUDA 12.x
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Configure settings:**
   - Edit `config.toml` to set data paths, training hyperparameters, and model paths
   - For CPU tuning: `python scripts/cpu_tuning.py` (recommends thread count)

### Basic Usage

**1. Gather and prepare data:**
```bash
# Download/generate BTCUSDT 30m data and create windows
python scripts/gather_additional_data.py --source binance --limit 5000 --out data/dataset.npz

# Label windows with bull flag patterns
python scripts/label_windows.py --progress
```

**2. Train the model:**
```bash
python scripts/train_real.py
```

**3. Run inference:**
```bash
python scripts/infer_real.py
```

**4. Evaluate metrics:**
```bash
python -m models.cnn_bullflag.eval_metrics \
  --ytrue models/cnn_bullflag/y_true.npy \
  --yprobs models/cnn_bullflag/y_probs.npy \
  --output-dir models/cnn_bullflag

# View metrics tables
python -m models.cnn_bullflag.report_metrics
```

**5. Launch dashboard:**
```bash
python scripts/dashboard.py
# Open http://localhost:5000 in your browser
```

## Key Concepts

### Per-Head Metrics

The model outputs probabilities for four heads: `flag`, `breakout`, `retest`, `continuation`. Per-head metrics help identify which pattern stages the model performs well on.

**Metrics computed:**
- **TP/FP/TN/FN** — Confusion matrix counts
- **Precision** — TP / (TP + FP)
- **Recall** — TP / (TP + FN)
- **F1 Score** — Harmonic mean of precision and recall
- **Accuracy** — (TP + TN) / Total

**Multi-threshold evaluation:**
Metrics are computed at thresholds 0.3, 0.5, and 0.7 to find optimal decision boundaries.

**Per-regime breakdown:**
Metrics can be segmented by market regime (e.g., "trending_up", "choppy") to identify when the model performs best.

**Example output:**
```
Head         Th   Precision  Recall  F1     Accuracy
flag         0.5  0.78       0.62    0.69   0.85
breakout     0.5  0.81       0.55    0.65   0.87
retest       0.5  0.60       0.40    0.48   0.82
continuation 0.5  0.72       0.50    0.59   0.84
```

### Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a 1D importance profile showing which time steps (bars) the model focuses on when making predictions.

**How it works:**
1. Captures activations from the last convolutional layer
2. Computes gradients of the target head output w.r.t. activations
3. Weights channels by gradient importance
4. Creates a heatmap normalized to [0, 1]
5. Upsamples to original sequence length (200 bars)

**Usage:**
```python
from models.cnn_bullflag.gradcam import GradCAM1D

gradcam = GradCAM1D(model, target_layer_name="conv4", device="cpu")
cam = gradcam.generate_cam(x, target_head="sequence")  # Returns [200] tensor
```

The activation map is included in inference outputs and can be visualized as a heatmap overlay on price charts.

## Usage Guide

### Data Pipeline

**1. Gather data:**
```bash
# From Binance API (real data)
python scripts/gather_additional_data.py --source binance --limit 5000 --append

# Generate synthetic data
python scripts/gather_additional_data.py --source synthetic --limit 15000 --append

# Output: data/dataset.npz with shape [N, 13, 200]
```

**2. Label windows:**
```bash
python scripts/label_windows.py --progress
# Output: data/dataset_labeled.npz with X:[N,13,200] and y:[N,4]
```

**Dataset format:**
- **X**: `[N, 13, 200]` — N windows × 13 channels × 200 bars
- **y**: `[N, 4]` — Binary labels for [flag, breakout, retest, continuation]
- **Channels**: open, high, low, close, volume, ema20, ema50, rsi14, vol_std20, volume_delta, body_norm, upper_wick, lower_wick

**Labeling parameters:**
Configure in `config.toml` `[labeler]` section:
- `pattern_zone` — Last N bars where patterns must occur (default: 80)
- `impulse_min_move_pct` — Minimum impulse move percentage
- `flag_max_range_pct` — Maximum flag consolidation range
- `breakout_min_pct` — Minimum breakout percentage
- And more...

### Training

**Configure in `config.toml`:**
```toml
[data]
train_x = "data/dataset_labeled.npz"
train_y = "data/dataset_labeled.npz"
val_x = "data/dataset_labeled.npz"
val_y = "data/dataset_labeled.npz"

[train]
epochs = 20
batch_size = 64
learning_rate = 0.001

[checkpoint]
model_path = "models/cnn_bullflag/checkpoints/cnn_v1.pt"

[paths]
eval_dir = "models/cnn_bullflag"
```

**Run training:**
```bash
python scripts/train_real.py
```

**Outputs:**
- Model checkpoint: `models/cnn_bullflag/checkpoints/cnn_v1.pt`
- Training log: `models/cnn_bullflag/training_log.json`
- Validation arrays: `models/cnn_bullflag/y_true.npy`, `y_probs.npy`

### Inference

**Configure in `config.toml`:**
```toml
[inference]
input_dir = ""  # Directory with window files, or leave empty
price_csv = ""  # CSV file path (alternative to input_dir)
window_len = 200
stride = 1
symbol = "BTCUSDT"
timeframe = "30m"
output_jsonl = "models/cnn_bullflag/outputs.jsonl"
enable_gradcam = true
```

**Run inference:**
```bash
python scripts/infer_real.py
```

**Output format (JSONL):**
```json
{
  "window_id": "window_0001",
  "symbol": "BTCUSDT",
  "tf": "30m",
  "window_start_ts": 1234567890,
  "window_end_ts": 1234567890,
  "scores": {
    "flag_prob": 0.85,
    "breakout_prob": 0.72,
    "retest_prob": 0.45,
    "continuation_prob": 0.38,
    "sequence_score": 0.57
  },
  "activation_map": {
    "indices": [0, 1, 2, ..., 199],
    "intensities": [0.1, 0.2, 0.5, ..., 0.8]
  },
  "meta": {
    "model_version": "cnn_bullflag_multihead_v1"
  }
}
```

### Evaluation

**Compute metrics:**
```bash
python -m models.cnn_bullflag.eval_metrics \
  --ytrue models/cnn_bullflag/y_true.npy \
  --yprobs models/cnn_bullflag/y_probs.npy \
  --regimes models/cnn_bullflag/regimes.npy \
  --output-dir models/cnn_bullflag
```

**View metrics tables:**
```bash
python -m models.cnn_bullflag.report_metrics
```

**Outputs:**
- `models/cnn_bullflag/test_multihead_metrics.json`
- `models/cnn_bullflag/test_multihead_metrics_by_regime.json`

### Visualization

**Grad-CAM gallery:**
```bash
python scripts/plot_gallery.py \
  --jsonl models/cnn_bullflag/outputs.jsonl \
  --out-dir models/cnn_bullflag/plots/gallery \
  --top-n 12 \
  --sort-by sequence_score
```

**Preview labels:**
```bash
python scripts/preview_labels.py --num-samples 6
```

**ASCII activation preview:**
```bash
python scripts/gradcam_ascii.py
```

## Web Dashboard

The dashboard provides real-time visualization of training progress, label statistics, predictions, and Grad-CAM heatmaps.

**Start dashboard:**
```bash
python scripts/dashboard.py
# Open http://localhost:5000
```

**Features:**
- **Training Progress** — Loss curves and training metrics
- **Label Statistics** — Distribution of labels with prevalence percentages
- **Sample Predictions** — Top predictions sorted by sequence score
- **Grad-CAM Heatmaps** — Click any sample to view activation heatmap

**Configuration:**
- Reads from `config.toml` automatically
- Auto-refreshes every 5 seconds
- Custom port: `python scripts/dashboard.py --port 8080`

**Data requirements:**
- Training log: `{eval_dir}/training_log.json`
- Evaluation arrays: `{eval_dir}/y_true.npy`, `y_probs.npy`
- Inference outputs: JSONL file with predictions

## Architecture

### Repository Structure

```
TradingCNN/
├── models/cnn_bullflag/
│   ├── model.py              # Multi-head CNN architecture
│   ├── gradcam.py            # Grad-CAM 1D implementation
│   ├── infer.py               # Inference wrapper with Grad-CAM
│   ├── eval_metrics.py       # Metrics computation
│   ├── report_metrics.py     # CLI for printing metrics tables
│   └── schema.py             # JSON output schema builder
├── scripts/
│   ├── train_real.py         # Training script
│   ├── infer_real.py          # Inference script
│   ├── label_windows.py      # Rule-based labeling pipeline
│   ├── gather_additional_data.py  # Data collection script
│   ├── dashboard.py           # Web dashboard server
│   ├── preview_labels.py     # Label visualization
│   └── plot_gallery.py       # Grad-CAM gallery generator
├── utils/
│   └── metrics.py            # Per-head metrics utilities
├── config.toml               # Configuration file
└── requirements.txt          # Python dependencies
```

### Model Architecture

- **Input**: `[B, 13, 200]` — Batch of windows with 13 channels and 200 time steps
- **Architecture**: 1D CNN with multiple convolutional blocks
- **Output**: Four heads (flag, breakout, retest, continuation) with sigmoid activations
- **Grad-CAM target**: Last convolutional layer (`conv4`)

### Data Flow

1. **Raw data** → OHLCV candles (BTCUSDT 30m)
2. **Feature engineering** → Compute technical indicators (EMA, RSI, volatility)
3. **Windowing** → Sliding windows of 200 bars
4. **Labeling** → Rule-based detection of bull flag patterns
5. **Training** → Multi-head binary classification
6. **Inference** → Generate predictions + Grad-CAM heatmaps
7. **Evaluation** → Per-head metrics and visualization

## Configuration

All settings are in `config.toml`:

```toml
[system]
threads = 6              # CPU threads (0 = auto-detect)
show_progress = true

[train]
epochs = 20
batch_size = 64
learning_rate = 0.001

[data]
train_x = "data/dataset_labeled.npz"
train_y = "data/dataset_labeled.npz"
val_x = "data/dataset_labeled.npz"
val_y = "data/dataset_labeled.npz"

[labeler]
pattern_zone = 80
impulse_min_move_pct = 0.05
flag_max_range_pct = 0.02
breakout_min_pct = 0.015
# ... more parameters

[inference]
window_len = 200
stride = 1
enable_gradcam = true
output_jsonl = "models/cnn_bullflag/outputs.jsonl"
```

## Troubleshooting

**Import errors:**
- Run scripts from project root: `python scripts/train_real.py`
- Set `PYTHONPATH=.` if needed: `export PYTHONPATH=.`

**Grad-CAM returns flat/near-zero maps:**
- Expected for random/untrained weights
- Train the model first or use trained checkpoint
- Check that target layer name matches (`conv4`)

**Metrics look odd:**
- Tune thresholds per head based on validation set
- Use `scripts/inspect_metrics.py` for calibration analysis
- Check label distribution: `python scripts/label_windows.py`

**Performance issues:**
- Grad-CAM adds backward pass overhead — disable for full sweeps
- Use CPU tuning: `python scripts/cpu_tuning.py`
- Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` environment variables

**No positive labels:**
- Adjust `[labeler]` parameters in `config.toml`
- Check label report: `models/cnn_bullflag/label_report.txt`
- Target prevalence: 1-5% per head

**Dashboard not showing data:**
- Ensure training has run: `python scripts/train_real.py`
- Check eval directory path in `config.toml`
- Verify inference outputs exist: `models/cnn_bullflag/outputs.jsonl`

## Requirements

- Python 3.9+
- NumPy 2.0+
- PyTorch 2.x (CPU or CUDA)
- Flask (for dashboard)
- pandas (for data processing)
- matplotlib (for visualization)

Install with:
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or cu121 for GPU
```

## Next Steps

- **Improve labeling**: Tune `[labeler]` parameters to achieve 1-5% prevalence per head
- **Train model**: Replace model stub with full architecture and train on labeled data
- **Tune thresholds**: Use validation metrics to find optimal decision thresholds per head
- **Extend metrics**: Add ROC-AUC, PR-AUC, calibration plots
- **Production integration**: Store outputs in datastore (Parquet/SQLite) and overlay on charts

## References

- **Per-head metrics**: See `utils/metrics.py` for implementation
- **Grad-CAM**: See `models/cnn_bullflag/gradcam.py` for 1D implementation
- **Labeling spec**: See `config.toml` `[labeler]` section for all parameters
- **Sequence score**: Geometric mean of all four head probabilities
