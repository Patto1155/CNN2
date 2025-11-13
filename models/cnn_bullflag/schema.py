from typing import Any, Dict, Optional, Sequence


def _to_list_floats(x: Optional[Sequence[float]]) -> Optional[list]:
    if x is None:
        return None
    return [float(v) for v in x]


def build_cnn_output(
    window_id: str,
    symbol: str,
    tf: str,
    start_ts: int,
    end_ts: int,
    infer_result: Dict[str, Any],
    model_version: str = "cnn_bullflag_multihead_v1",
) -> Dict[str, Any]:
    """
    Compose a JSON-serializable output dict from an inference result.

    Expected infer_result keys:
      - flag_prob, breakout_prob, retest_prob, continuation_prob, sequence_score
      - activation_map: list[float] or None (length == seq_len)
    """
    activation_map = infer_result.get("activation_map")
    activation_struct: Optional[Dict[str, Any]]
    if activation_map is None:
        activation_struct = None
    else:
        values = _to_list_floats(activation_map)
        activation_struct = {
            "indices": list(range(len(values))),  # type: ignore[arg-type]
            "intensities": values,
        }

    return {
        "window_id": window_id,
        "symbol": symbol,
        "tf": tf,
        "window_start_ts": int(start_ts),
        "window_end_ts": int(end_ts),
        "scores": {
            "flag_prob": float(infer_result.get("flag_prob", 0.0)),
            "breakout_prob": float(infer_result.get("breakout_prob", 0.0)),
            "retest_prob": float(infer_result.get("retest_prob", 0.0)),
            "continuation_prob": float(infer_result.get("continuation_prob", 0.0)),
            "sequence_score": float(infer_result.get("sequence_score", 0.0)),
        },
        "activation_map": activation_struct,
        "meta": {
            "model_version": model_version,
        },
    }

