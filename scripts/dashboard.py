#!/usr/bin/env python3
"""
Web dashboard for visualizing CNN training progress, CAM heatmaps, and predictions.
Run with: python scripts/dashboard.py
Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._setup_path import add_project_root

add_project_root()

try:
    from flask import Flask, jsonify, render_template_string, send_from_directory
except ImportError:
    print("Flask not installed. Install with: pip install flask")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

from scripts.config import load_config

app = Flask(__name__)

# Global state
CONFIG = None
EVAL_DIR = None
INFERENCE_OUTPUT = None


def load_training_log(eval_dir: Path) -> Optional[Dict]:
    """Load training log JSON."""
    log_path = eval_dir / "training_log.json"
    if not log_path.exists():
        return None
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_eval_arrays(eval_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load y_true and y_probs arrays."""
    y_true_path = eval_dir / "y_true.npy"
    y_probs_path = eval_dir / "y_probs.npy"
    
    if not (y_true_path.exists() and y_probs_path.exists()):
        return None
    
    try:
        return {
            "y_true": np.load(y_true_path),
            "y_probs": np.load(y_probs_path),
        }
    except Exception:
        return None


def load_inference_outputs(output_path: Path) -> Optional[List[Dict]]:
    """Load inference JSONL outputs."""
    if not output_path.exists():
        return None
    
    try:
        rows = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except Exception:
        return None


def compute_label_stats(y_true: np.ndarray, y_probs: np.ndarray) -> Dict:
    """Compute label distribution statistics."""
    label_names = ["flag", "breakout", "retest", "continuation"]
    stats = {}
    
    for i, name in enumerate(label_names):
        true_positives = int((y_true[:, i] == 1).sum())
        total = len(y_true)
        stats[name] = {
            "positives": true_positives,
            "total": total,
            "prevalence": float(true_positives / total * 100) if total > 0 else 0.0,
            "mean_prob": float(y_probs[:, i].mean()) if len(y_probs) > 0 else 0.0,
        }
    
    return stats


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/training_log")
def api_training_log():
    """Get training log data."""
    if EVAL_DIR is None:
        return jsonify({"error": "No eval directory configured"}), 404
    
    log = load_training_log(EVAL_DIR)
    if log is None:
        return jsonify({"error": "Training log not found"}), 404
    
    return jsonify(log)


@app.route("/api/label_stats")
def api_label_stats():
    """Get label distribution statistics."""
    if EVAL_DIR is None:
        return jsonify({"error": "No eval directory configured"}), 404
    
    arrays = load_eval_arrays(EVAL_DIR)
    if arrays is None:
        return jsonify({"error": "Evaluation arrays not found"}), 404
    
    stats = compute_label_stats(arrays["y_true"], arrays["y_probs"])
    return jsonify(stats)


@app.route("/api/predictions")
def api_predictions():
    """Get sample predictions from inference outputs."""
    if INFERENCE_OUTPUT is None or not INFERENCE_OUTPUT.exists():
        return jsonify({"error": "Inference output not found"}), 404
    
    rows = load_inference_outputs(INFERENCE_OUTPUT)
    if rows is None:
        return jsonify({"error": "Could not load inference outputs"}), 404
    
    # Return top N by sequence score
    sorted_rows = sorted(rows, key=lambda r: r.get("scores", {}).get("sequence_score", 0), reverse=True)
    return jsonify({"samples": sorted_rows[:50]})


@app.route("/api/cam/<int:index>")
def api_cam(index: int):
    """Get CAM heatmap data for a specific sample."""
    if INFERENCE_OUTPUT is None or not INFERENCE_OUTPUT.exists():
        return jsonify({"error": "Inference output not found"}), 404
    
    rows = load_inference_outputs(INFERENCE_OUTPUT)
    if rows is None or index >= len(rows):
        return jsonify({"error": "Sample not found"}), 404
    
    row = rows[index]
    activation_map = row.get("activation_map")
    if activation_map is None:
        return jsonify({"error": "No activation map for this sample"}), 404
    
    return jsonify({
        "indices": activation_map.get("indices", []),
        "intensities": activation_map.get("intensities", []),
        "scores": row.get("scores", {}),
    })


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CNN Bull Flag Dashboard</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #4a9eff; margin-bottom: 30px; }
        .section {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .section h2 { color: #6bb6ff; margin-bottom: 15px; font-size: 1.3em; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-card {
            background: #333;
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #4a9eff;
        }
        .stat-label { color: #aaa; font-size: 0.9em; }
        .stat-value { color: #4a9eff; font-size: 1.5em; font-weight: bold; margin-top: 5px; }
        .chart-container {
            background: #1e1e1e;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
            height: 300px;
        }
        .samples-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .sample-item {
            background: #333;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .sample-item:hover { background: #3a3a3a; }
        .sample-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .sample-scores {
            display: flex;
            gap: 15px;
            font-size: 0.9em;
        }
        .score-item { color: #aaa; }
        .score-value { color: #6bb6ff; font-weight: bold; }
        .cam-heatmap {
            margin-top: 20px;
            padding: 15px;
            background: #1e1e1e;
            border-radius: 6px;
        }
        .heatmap-bar {
            height: 40px;
            background: linear-gradient(to right, #000 0%, #ff4500 50%, #ffff00 100%);
            border-radius: 4px;
            margin-top: 10px;
            position: relative;
        }
        .heatmap-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
        }
        .loading { color: #888; text-align: center; padding: 20px; }
        .error { color: #ff6b6b; padding: 10px; background: #3a1f1f; border-radius: 4px; }
        button {
            background: #4a9eff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover { background: #5aaeff; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>🚀 CNN Bull Flag Training Dashboard</h1>
        
        <div class="section">
            <h2>Training Progress</h2>
            <div id="training-info" class="loading">Loading training log...</div>
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>
        
        <div class="section">
            <h2>Label Statistics</h2>
            <div id="label-stats" class="loading">Loading label statistics...</div>
            <div class="stats-grid" id="stats-grid"></div>
        </div>
        
        <div class="section">
            <h2>Sample Predictions</h2>
            <div id="samples-list" class="samples-list loading">Loading predictions...</div>
        </div>
        
        <div class="section" id="cam-section" style="display: none;">
            <h2>Grad-CAM Heatmap</h2>
            <div id="cam-heatmap" class="cam-heatmap"></div>
        </div>
    </div>
    
    <script>
        let lossChart = null;
        let selectedSampleIndex = null;
        
        async function loadTrainingLog() {
            try {
                const res = await fetch('/api/training_log');
                if (!res.ok) {
                    document.getElementById('training-info').innerHTML = '<div class="error">Training log not available</div>';
                    return;
                }
                const data = await res.json();
                document.getElementById('training-info').innerHTML = `
                    <div>Epochs: ${data.epochs || 'N/A'} | Batch Size: ${data.batch_size || 'N/A'} | LR: ${data.lr || 'N/A'}</div>
                `;
                
                if (data.train_losses && data.train_losses.length > 0) {
                    const ctx = document.getElementById('lossChart').getContext('2d');
                    lossChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.train_losses.map((_, i) => i + 1),
                            datasets: [{
                                label: 'Train Loss',
                                data: data.train_losses,
                                borderColor: '#4a9eff',
                                backgroundColor: 'rgba(74, 158, 255, 0.1)',
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { beginAtZero: true, ticks: { color: '#aaa' }, grid: { color: '#333' } },
                                x: { ticks: { color: '#aaa' }, grid: { color: '#333' } }
                            },
                            plugins: {
                                legend: { labels: { color: '#aaa' } }
                            }
                        }
                    });
                }
            } catch (e) {
                document.getElementById('training-info').innerHTML = '<div class="error">Error loading training log</div>';
            }
        }
        
        async function loadLabelStats() {
            try {
                const res = await fetch('/api/label_stats');
                if (!res.ok) {
                    document.getElementById('label-stats').innerHTML = '<div class="error">Label stats not available</div>';
                    return;
                }
                const stats = await res.json();
                document.getElementById('label-stats').innerHTML = '';
                
                const grid = document.getElementById('stats-grid');
                grid.innerHTML = '';
                for (const [name, data] of Object.entries(stats)) {
                    const card = document.createElement('div');
                    card.className = 'stat-card';
                    card.innerHTML = `
                        <div class="stat-label">${name}</div>
                        <div class="stat-value">${data.positives.toLocaleString()}</div>
                        <div style="color: #888; font-size: 0.85em; margin-top: 5px;">
                            ${data.prevalence.toFixed(2)}% prevalence | Mean prob: ${data.mean_prob.toFixed(3)}
                        </div>
                    `;
                    grid.appendChild(card);
                }
            } catch (e) {
                document.getElementById('label-stats').innerHTML = '<div class="error">Error loading label stats</div>';
            }
        }
        
        async function loadSamples() {
            try {
                const res = await fetch('/api/predictions');
                if (!res.ok) {
                    document.getElementById('samples-list').innerHTML = '<div class="error">Predictions not available</div>';
                    return;
                }
                const data = await res.json();
                const samples = data.samples || [];
                
                const list = document.getElementById('samples-list');
                list.innerHTML = '';
                list.className = 'samples-list';
                
                samples.forEach((sample, idx) => {
                    const item = document.createElement('div');
                    item.className = 'sample-item';
                    const scores = sample.scores || {};
                    item.innerHTML = `
                        <div class="sample-header">
                            <div><strong>Sample ${idx}</strong></div>
                            <div class="sample-scores">
                                <span class="score-item">Seq: <span class="score-value">${(scores.sequence_score || 0).toFixed(3)}</span></span>
                                <span class="score-item">Flag: <span class="score-value">${(scores.flag_prob || 0).toFixed(3)}</span></span>
                                <span class="score-item">Breakout: <span class="score-value">${(scores.breakout_prob || 0).toFixed(3)}</span></span>
                                <span class="score-item">Retest: <span class="score-value">${(scores.retest_prob || 0).toFixed(3)}</span></span>
                                <span class="score-item">Cont: <span class="score-value">${(scores.continuation_prob || 0).toFixed(3)}</span></span>
                            </div>
                        </div>
                    `;
                    item.onclick = () => loadCAM(idx);
                    list.appendChild(item);
                });
            } catch (e) {
                document.getElementById('samples-list').innerHTML = '<div class="error">Error loading predictions</div>';
            }
        }
        
        async function loadCAM(index) {
            try {
                const res = await fetch(`/api/cam/${index}`);
                if (!res.ok) {
                    return;
                }
                const data = await res.json();
                selectedSampleIndex = index;
                
                document.getElementById('cam-section').style.display = 'block';
                const heatmapDiv = document.getElementById('cam-heatmap');
                
                const intensities = data.intensities || [];
                const indices = data.indices || [];
                
                if (intensities.length === 0) {
                    heatmapDiv.innerHTML = '<div class="error">No activation map data</div>';
                    return;
                }
                
                let html = `<h3>Sample ${index} - Sequence Score: ${(data.scores?.sequence_score || 0).toFixed(3)}</h3>`;
                html += '<div class="heatmap-bar">';
                
                // Create gradient overlay based on intensities
                const maxIntensity = Math.max(...intensities);
                intensities.forEach((intensity, i) => {
                    const opacity = intensity / maxIntensity;
                    const left = (i / intensities.length) * 100;
                    const width = (1 / intensities.length) * 100;
                    html += `<div style="position: absolute; left: ${left}%; width: ${width}%; height: 100%; background: rgba(255, 69, 0, ${opacity});"></div>`;
                });
                
                html += '</div>';
                html += `<div style="margin-top: 10px; color: #888; font-size: 0.9em;">Max intensity: ${maxIntensity.toFixed(3)}</div>`;
                
                heatmapDiv.innerHTML = html;
            } catch (e) {
                console.error('Error loading CAM:', e);
            }
        }
        
        // Initial load
        loadTrainingLog();
        loadLabelStats();
        loadSamples();
        
        // Auto-refresh every 5 seconds
        setInterval(() => {
            loadTrainingLog();
            loadLabelStats();
            loadSamples();
        }, 5000);
    </script>
</body>
</html>
"""


def main() -> None:
    global CONFIG, EVAL_DIR, INFERENCE_OUTPUT
    
    parser = argparse.ArgumentParser(description="CNN Training Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--eval-dir", type=str, help="Override eval directory path")
    parser.add_argument("--inference-output", type=str, help="Path to inference JSONL output")
    args = parser.parse_args()
    
    # Load config
    CONFIG = load_config()
    
    # Set eval directory
    if args.eval_dir:
        EVAL_DIR = Path(args.eval_dir)
    else:
        eval_dir_str = CONFIG.get("paths", "eval_dir", default="models/cnn_bullflag")
        EVAL_DIR = Path(str(eval_dir_str))
    
    # Set inference output
    if args.inference_output:
        INFERENCE_OUTPUT = Path(args.inference_output)
    else:
        inference_output_str = CONFIG.get("inference", "output_jsonl", default="models/cnn_bullflag/outputs.jsonl")
        INFERENCE_OUTPUT = Path(str(inference_output_str))
    
    print(f"Dashboard starting on http://{args.host}:{args.port}")
    print(f"Eval directory: {EVAL_DIR}")
    print(f"Inference output: {INFERENCE_OUTPUT}")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

