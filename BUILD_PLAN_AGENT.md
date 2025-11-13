# Agent Build Plan (Tick-Box)

Sequence of tasks for the BullFlag CNN metrics + Grad-CAM project. All boxes are now checked with references to the delivered artifacts.

Legend: [ ] pending, [x] complete

---

## Phase 0 – Skeleton & Conventions

- [x] Create base module structure (packages ready: `models/cnn_bullflag`, `utils`).
- [x] Commit this build plan (`BUILD_PLAN_AGENT.md`).

Artifacts: package folders, this document.

---

## Phase 1 – Per-Head Metrics

- [x] Metrics utilities (`utils/metrics.py`).
- [x] Synthetic metrics smoke test (`scripts/test_metrics_synthetic.py`).
- [x] Evaluation integration (`models/cnn_bullflag/eval_metrics.py`, saves JSON to `models/cnn_bullflag/test_multihead_metrics*.json`).
- [x] Reporting CLI (`models/cnn_bullflag/report_metrics.py`).

Artifacts: metrics JSON files, reporting CLI output.

---

## Phase 2 – Grad-CAM 1D

- [x] GradCAM utility (`models/cnn_bullflag/gradcam.py`).
- [x] Model exposes `conv4` (stub in `models/cnn_bullflag/model_stub.py`).
- [x] Inference wiring with optional Grad-CAM (`models/cnn_bullflag/infer.py`).
- [x] Smoke test (`scripts/smoke_gradcam.py`).

Artifacts: activation arrays per window, smoke-test log.

---

## Phase 3 – Output Schema & Analytics

- [x] Schema builder (`models/cnn_bullflag/schema.py`).
- [x] Wrap inference outputs (`scripts/generate_cnn_output_demo.py`, `run_inference_batch`).
- [x] Visualization/analytics hooks (`scripts/analytics_demo.py`).

Artifacts: `example_cnn_outputs.jsonl`, analytics summaries.

---

## Phase 4 – Validation & Iteration

- [x] Full train/test cycle (simulated) – `scripts/full_cycle_mock.py`.
- [x] Metrics & calibration inspection – `scripts/inspect_metrics.py`.
- [x] Grad-CAM spot checks – `scripts/gradcam_ascii.py`.
- [x] Sequencing metric decision – `sequence_score_decision.md`.

Artifacts: console logs, decision note.

---

## Reference Paths & Commands

- Metrics JSON: `models/cnn_bullflag/test_multihead_metrics.json` and `_by_regime.json`.
- Grad-CAM smoke test: `python scripts/smoke_gradcam.py`.
- Metrics CLI: `python -m models.cnn_bullflag.report_metrics`.
- Demo inference dump: `python scripts/generate_cnn_output_demo.py`.

With every checkbox satisfied, Cline (or any agent) can now rerun scripts or extend functionality incrementally.

