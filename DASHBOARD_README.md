# CNN Bull Flag Dashboard

A web-based dashboard for visualizing CNN training progress, Grad-CAM heatmaps, and predictions.

## Quick Start

1. **Start the dashboard server:**
   ```bash
   python scripts/dashboard.py
   ```

2. **Open in browser:**
   ```
   http://localhost:5000
   ```

## Features

- **Training Progress**: View loss curves and training metrics from `training_log.json`
- **Label Statistics**: See distribution of labels (flag, breakout, retest, continuation) with prevalence percentages
- **Sample Predictions**: Browse top predictions from inference outputs, sorted by sequence score
- **Grad-CAM Heatmaps**: Click any sample to view its activation heatmap showing which time steps the model focuses on

## Configuration

The dashboard automatically reads from `config.toml`:
- **Eval directory**: `[paths].eval_dir` (default: `models/cnn_bullflag`)
- **Inference output**: `[inference].output_jsonl` (default: `models/cnn_bullflag/outputs.jsonl`)

You can override these with CLI arguments:
```bash
python scripts/dashboard.py --eval-dir path/to/eval --inference-output path/to/outputs.jsonl --port 8080
```

## Data Requirements

The dashboard expects:
1. **Training log**: `{eval_dir}/training_log.json` (created by `scripts/train_real.py`)
2. **Evaluation arrays**: `{eval_dir}/y_true.npy` and `{eval_dir}/y_probs.npy` (created by training)
3. **Inference outputs**: JSONL file with predictions and activation maps (created by `scripts/infer_real.py`)

## Auto-Refresh

The dashboard automatically refreshes every 5 seconds to show latest training progress.

## Troubleshooting

- **"Training log not available"**: Run training first with `python scripts/train_real.py`
- **"Predictions not available"**: Run inference first with `python scripts/infer_real.py`
- **Port already in use**: Use `--port` to specify a different port

