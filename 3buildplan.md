3. Build plan for Cline (concrete tasks)

Here’s how I’d hand this to Cline as a sequence of implementation steps.

Phase 1 – Metrics module (per-head evaluation)

Goal: Be able to say “CNN is good at flags, bad at retests, especially in chop.”

Tasks:

Create utils/metrics.py

Implement:

compute_binary_metrics(y_true, y_prob, threshold)

compute_multihead_metrics(y_true, y_probs, thresholds=(0.3,0.5,0.7))

compute_multihead_metrics_by_regime(y_true, y_probs, regimes, thresholds=(0.5,))

Add unit tests with synthetic data to validate metrics.

Integrate into train.py

After training, when evaluating on the test set:

Gather:

y_true_all: [N, 4]

y_prob_all: [N, 4]

optional regimes_all: [N]

Call compute_multihead_metrics and compute_multihead_metrics_by_regime.

Save results to:

models/cnn_bullflag/test_multihead_metrics.json

models/cnn_bullflag/test_multihead_metrics_by_regime.json

Add a simple CLI / script

python -m models.cnn_bullflag.report_metrics

This prints a table like:

Head       Th   Precision  Recall  F1
flag       0.5  0.78       0.62    0.69
breakout   0.5  0.81       0.55    0.65
retest     0.5  0.60       0.40    0.48
continuation 0.5 0.72      0.50    0.59

Phase 2 – 1D Grad-CAM integration

Goal: For any window, get a [seq_len] importance profile for visualisation.

Tasks:

Add models/cnn_bullflag/gradcam.py

Implement GradCAM1D exactly as specced:

Constructor wires to target layer (e.g. "conv4").

Registers forward + backward hooks.

generate_cam(x, target_head="sequence") returns [seq_len] tensor in [0,1].

Modify BullFlagCNNMultiHead

Ensure last conv block is named (e.g.) self.conv4 so GradCAM can target it by name.

Confirm the forward passes through conv4 before pooling.

Extend infer.py

Add optional enable_gradcam flag.

When enabled:

Perform Grad-CAM call for target_head="sequence" and capture cam_upsampled (length 200).

Include it in the output dict under "activation_map".

Add quick smoke test

Create a random input [1, 12, 200], run Grad-CAM, confirm:

No crash, output shape [200]

Values in [0, 1]

Optionally log a few values.

Phase 3 – Wire into CNNOutput JSON + analytics

Goal: Make the outputs consumable by your app + visualisation layer.

Tasks:

Define CNNOutput schema in one place

e.g. models/cnn_bullflag/schema.py:

def build_cnn_output(window_id, symbol, tf, start_ts, end_ts, infer_result):
    return {
        "window_id": window_id,
        "symbol": symbol,
        "tf": tf,
        "window_start_ts": start_ts,
        "window_end_ts": end_ts,
        "scores": {
            "flag_prob": infer_result["flag_prob"],
            "breakout_prob": infer_result["breakout_prob"],
            "retest_prob": infer_result["retest_prob"],
            "continuation_prob": infer_result["continuation_prob"],
            "sequence_score": infer_result["sequence_score"],
        },
        "activation_map": (
            None if infer_result["activation_map"] is None
            else {
                "indices": list(range(len(infer_result["activation_map"]))),
                "intensities": infer_result["activation_map"],
            }
        ),
        "meta": {
            "model_version": "cnn_bullflag_multihead_v1"
        }
    }


Update the engine’s CNN worker

Whenever you run inference on a window:

Call BullFlagCNNInfer(...) to get infer_result.

Wrap with build_cnn_output(...).

Store in your cnn_signals datastore (parquet/sqlite/etc).

Update analytics / pattern gallery code

Use scores.sequence_score to:

Rank windows.

Select top N per head (e.g. top flag_prob examples, top retest failures, etc.).

Use activation_map to show:

Which bars inside each window got the most attention.

Heat overlays under the price chart.

Phase 4 – Validate end-to-end

Goal: Sanity-check that everything works and is useful.

Tasks:

Run a full training cycle.

Run test evaluation; inspect:

test_multihead_metrics.json

test_multihead_metrics_by_regime.json

Take a handful of high sequence_score windows:

Use Grad-CAM to visualise them on a chart (quick hack script with matplotlib is fine).

Confirm by eye: activation is mostly on flag + breakout + retest, not random noise.

Take a handful of false-positives and false-negatives and inspect:

Are thresholds wrong?

Is the labelling spec too strict or too lenient?

Is one head clearly underperforming?

From there you’ll know exactly whether:

the model actually understands the pattern, and

which phase (flag / breakout / retest / continuation) needs more work or better labels.