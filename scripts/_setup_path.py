"""Ensure the repository root is on sys.path for standalone scripts."""

import sys
from pathlib import Path


def add_project_root() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

