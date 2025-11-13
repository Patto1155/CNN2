# Sequence Score Aggregation Decision

Purpose: choose a deterministic way to aggregate multihead probabilities into a single `sequence_score` used for ranking candidate windows.

## Options considered

1. **Minimum head probability**: harshly penalizes any weak head; prioritizes balanced patterns but suppresses mixed signals.
2. **Arithmetic mean**: smooths variance but can overrate windows where one head is extremely confident while others are noisy.
3. **Geometric mean (current approach)**: rewards consistently high heads while still differentiating mixed-quality windows; sensitive to zeros but stable with epsilon.
4. **Weighted mean**: same as arithmetic/geometric but with per-head weights derived from validation metrics.

## Decision

- **Adopt the geometric mean** across the four head probabilities as `sequence_score`.
- Rationale: encourages windows where every phase (flag, breakout, retest, continuation) aligns; easy to interpret and already used inside `GradCAM1D`.
- Implementation: see `models/cnn_bullflag/infer.py::_geometric_mean`.
- Next steps: update the analytics scripts (`scripts/analytics_demo.py`) to expose per-head rankings when investigating specific failure modes.

