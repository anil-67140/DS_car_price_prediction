"""Raw data acquisition: local file -> URL -> Kaggle API. Records provenance and licence."""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings
from ..utils.io import atomic_path, read_json, sha256_file, write_json

log = logging.getLogger("carprice.ingest")


class DatasetUnavailableError(RuntimeError):
    pass


def _find_local(settings: Settings, src: dict[str, Any]) -> Path | None:
    for cand in [settings.paths.data_raw / src["filename"], *src.get("local_candidates", [])]:
        p = Path(cand)
        if not p.is_absolute():
            p = settings.paths.home / p
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _download_url(url: str, dest: Path) -> None:
    log.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "carprice-pipeline/1.0"})
    with atomic_path(dest) as tmp, urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)


def _download_kaggle(slug: str, filename: str, dest_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Download via the Kaggle API. Credentials come ONLY from the environment / kaggle.json.

    Not exercised in the sandbox this repository snapshot was authored in (kaggle.com is
    unreachable there); it runs in Colab. Failures raise a clear error with setup steps.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover - depends on env
        raise DatasetUnavailableError(f"kaggle package not importable: {exc}") from exc
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover
        raise DatasetUnavailableError(
            "Kaggle authentication failed. Provide KAGGLE_USERNAME/KAGGLE_KEY env vars or "
            "~/.kaggle/kaggle.json (see README: 'Kaggle API setup')."
        ) from exc
    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(dest_dir), unzip=False, quiet=True)
    zips = sorted(dest_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise DatasetUnavailableError(f"Kaggle download of {slug} produced no archive")
    with zipfile.ZipFile(zips[-1]) as zf:
        zf.extract(filename, dest_dir)
    zips[-1].unlink()
    meta: dict[str, Any] = {}
    try:  # licence as declared by Kaggle at download time
        api.dataset_metadata(slug, path=str(dest_dir))
        meta = read_json(dest_dir / "dataset-metadata.json")
    except Exception as exc:  # pragma: no cover
        log.warning("Could not read Kaggle metadata: %s", exc)
    return dest_dir / filename, meta


def fetch_raw(settings: Settings) -> tuple[Path, dict[str, Any]]:
    """Return (path to raw file, provenance dict). Never silently falls back to other data."""
    src = settings.dataset["source"]
    dest_dir = settings.paths.data_raw
    dest_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {}

    local = _find_local(settings, src)
    if local is not None:
        path = local
        provenance["acquired_via"] = "local_file"
    elif src["type"] == "url":
        path = dest_dir / src["filename"]
        _download_url(src["url"], path)
        provenance.update(acquired_via="url", url=src["url"])
    elif src["type"] == "local":
        raise DatasetUnavailableError(f"Expected local file {dest_dir / src['filename']} (type: local) but it does not exist")
    elif src["type"] == "kaggle":
        path, meta = _download_kaggle(src["slug"], src["filename"], dest_dir)
        provenance.update(acquired_via="kaggle_api", slug=src["slug"], kaggle_metadata=meta)
    else:
        raise DatasetUnavailableError(f"Unknown source type {src['type']!r}")

    if path.parent != dest_dir:  # copy into the project's raw dir so it lives next to artifacts
        target = dest_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)
        path = target

    provenance.update(
        dataset=settings.dataset_key,
        filename=path.name,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        retrieved_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        declared_license=settings.dataset.get("license", "").strip(),
        pipeline_verified_on_this_dataset=bool(settings.dataset.get("verified", False)),
    )
    return path, provenance


def read_raw(path: Path) -> pd.DataFrame:
    # read everything as text: parsing/typing is the cleaning stage's job and is reported there
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def write_provenance(settings: Settings, provenance: dict[str, Any], n_rows: int) -> Path:
    out = settings.paths.reports / "dataset_sources.json"
    existing: dict[str, Any] = read_json(out) if out.exists() else {}
    existing[settings.dataset_key] = {**provenance, "raw_rows": n_rows}
    write_json(out, existing)
    return out
