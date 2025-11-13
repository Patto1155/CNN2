"""Small utilities to download files with a progress bar."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def download_file(url: str, dest: Path, chunk_size: int = 8192, show_progress: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        if show_progress and tqdm is not None and total > 0:
            bar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}")
        else:
            bar = None
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if bar is not None:
            bar.close()
    return dest

