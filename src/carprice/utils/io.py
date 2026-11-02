"""Atomic, verifiable file I/O.

Every write goes to a temporary file in the SAME directory, is flushed + fsync'd, and is then
moved into place with ``os.replace`` (atomic on POSIX and on Google Drive's FUSE mount in practice).
A runtime that is killed mid-write therefore leaves either the old file or the complete new file,
never a truncated one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(obj: Any) -> str:
    """Deterministic short hash of any JSON-serialisable object (used for config hashing)."""
    payload = json.dumps(obj, sort_keys=True, default=str)
    return sha256_text(payload)[:16]


@contextlib.contextmanager
def atomic_path(final_path: str | Path) -> Iterator[Path]:
    """Yield a temp path next to ``final_path``; atomically move it into place on success."""
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=final_path.parent, prefix=f".{final_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        yield tmp_path
        with open(tmp_path, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def atomic_write(final_path: str | Path, writer: Callable[[Path], None]) -> Path:
    with atomic_path(final_path) as tmp:
        writer(tmp)
    return Path(final_path)


def write_text(path: str | Path, text: str) -> Path:
    return atomic_write(path, lambda p: p.write_text(text, encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> Path:
    return write_text(path, json.dumps(obj, indent=2, default=str, sort_keys=True))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, df: pd.DataFrame, index: bool = False) -> Path:
    return atomic_write(path, lambda p: df.to_csv(p, index=index))


def write_parquet_or_csv(path: str | Path, df: pd.DataFrame) -> Path:
    """Parquet if an engine is installed, otherwise CSV (same stem)."""
    path = Path(path)
    try:
        import pyarrow  # noqa: F401

        return atomic_write(path.with_suffix(".parquet"), lambda p: df.to_parquet(p, index=False))
    except ImportError:
        return write_csv(path.with_suffix(".csv"), df)


def dump_joblib(path: str | Path, obj: Any) -> Path:
    return atomic_write(path, lambda p: joblib.dump(obj, p, compress=3))


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def save_figure(fig, path: str | Path, dpi: int = 130) -> Path:
    def _w(p: Path) -> None:
        fig.savefig(p, dpi=dpi, bbox_inches="tight", format=Path(path).suffix.lstrip(".") or "png")

    return atomic_write(path, _w)


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:,.3f}", max_rows: int | None = None) -> str:
    """Dependency-free Markdown table."""
    d = df if max_rows is None else df.head(max_rows)
    cols = list(d.columns)

    def fmt(v):
        if isinstance(v, float):
            return "" if v != v else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)
