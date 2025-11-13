"""Lightweight configuration loader for project settings (TOML).

Reads `config.toml` from the project root and exposes a dict-like object.
Uses Python 3.11+ `tomllib` (available in 3.13). No extra deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:  # Python 3.11+
    import tomllib as toml
except Exception:  # pragma: no cover
    import tomli as toml  # type: ignore


DEFAULTS: Dict[str, Any] = {
    "system": {"threads": 0, "show_progress": True},
    "train": {"steps": 50},
    "evaluate": {"enable_gradcam": True},
    "inference": {"output_jsonl": "models/cnn_bullflag/example_cnn_outputs.jsonl", "enable_gradcam": True},
    "gallery": {"top_n": 12, "sort_by": "sequence_score", "output_dir": "models/cnn_bullflag/plots/gallery"},
}


@dataclass
class Config:
    data: Dict[str, Any]

    def get(self, *path: str, default: Any | None = None) -> Any:
        cur: Any = self.data
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[index]
        else:
            out[k] = v
    return out


def load_config(path: Path | None = None) -> Config:
    if path is None:
        path = Path("config.toml")
    if path.exists():
        with path.open("rb") as fh:
            user_cfg = toml.load(fh)
        data = _deep_merge(DEFAULTS, user_cfg)
    else:
        data = DEFAULTS
    return Config(data=data)

