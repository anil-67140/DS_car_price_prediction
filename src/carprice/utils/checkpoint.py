"""Experiment manifest + resumable stages.

Design
------
* The manifest (``artifacts/checkpoints/manifest.json``) is the single source of truth about
  which units of work are finished. It is itself written atomically.
* A unit of work (a *stage* such as ``clean`` or an *experiment* such as ``model::ridge``) is
  recorded only AFTER all its output files exist. Each entry stores: config hash, input hash,
  output files with size + sha256, metrics, timestamps and software versions.
* A unit is skipped on re-run only if (a) the manifest says ``complete``, (b) the config hash and
  upstream input hash still match, and (c) every output file still exists with the recorded sha256.
  Otherwise it is recomputed. This is what "verify saved files before reusing them" means here.
* Nothing relies on Python variables surviving a Colab disconnect - everything reloads from disk.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json

log = logging.getLogger("carprice.checkpoint")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Manifest:
    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        path = Path(path)
        if path.exists():
            try:
                data = read_json(path)
                return cls(path=path, entries=data.get("entries", {}))
            except Exception as exc:  # corrupt manifest must never block the run
                log.warning("Manifest unreadable (%s); starting a fresh one. Old file kept.", exc)
                path.rename(path.with_suffix(f".corrupt-{int(time.time())}.json"))
        return cls(path=path)

    def save(self) -> None:
        write_json(self.path, {"schema": 1, "updated": _now(), "entries": self.entries})

    # ---- verification -------------------------------------------------------------------------
    def verify(self, key: str, config_hash: str, input_hash: str = "") -> tuple[bool, str]:
        e = self.entries.get(key)
        if not e:
            return False, "no manifest entry"
        if e.get("status") != "complete":
            return False, f"status={e.get('status')}"
        if e.get("config_hash") != config_hash:
            return False, "config changed"
        if e.get("input_hash", "") != input_hash:
            return False, "upstream input changed"
        for rel, meta in e.get("outputs", {}).items():
            p = Path(rel)
            if not p.is_absolute():
                p = self.path.parent.parent.parent / rel
            if not p.exists():
                return False, f"missing output {rel}"
            if p.stat().st_size != meta["size"] or sha256_file(p) != meta["sha256"]:
                return False, f"output changed/corrupt {rel}"
        return True, "verified"

    def record(
        self,
        key: str,
        outputs: list[Path],
        config_hash: str,
        input_hash: str = "",
        metrics: dict[str, Any] | None = None,
        seconds: float | None = None,
        extra: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> None:
        root = root or self.path.parent.parent.parent
        outs = {}
        for p in outputs:
            p = Path(p)
            try:
                rel = str(p.resolve().relative_to(root.resolve()))
            except ValueError:
                rel = str(p.resolve())
            outs[rel] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
        self.entries[key] = {
            "status": "complete",
            "config_hash": config_hash,
            "input_hash": input_hash,
            "outputs": outs,
            "metrics": metrics or {},
            "seconds": None if seconds is None else round(seconds, 2),
            "finished": _now(),
            "extra": extra or {},
        }
        self.save()

    def mark_failed(self, key: str, error: str) -> None:
        self.entries[key] = {"status": "failed", "error": error[:500], "finished": _now()}
        self.save()

    def output_hash(self, key: str) -> str:
        """Hash of the recorded outputs of ``key`` - used as the input hash of downstream stages."""
        e = self.entries.get(key, {})
        from .io import stable_hash

        return stable_hash(e.get("outputs", {}))


def run_unit(
    manifest: Manifest,
    key: str,
    fn: Callable[[], tuple[list[Path], dict[str, Any]]],
    config_hash: str,
    input_hash: str = "",
    force: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Run ``fn`` unless a verified completed result exists.

    ``fn`` must write its outputs (atomically) and return ``(output_paths, metrics)``.
    Returns the manifest entry. ``entry['skipped']`` is True when the cache was reused.
    """
    if not force:
        ok, why = manifest.verify(key, config_hash, input_hash)
        if ok:
            log.info("SKIP  %-34s (verified, finished %s)", key, manifest.entries[key]["finished"])
            return {**manifest.entries[key], "skipped": True}
        log.info("RUN   %-34s (%s)", key, why)
    else:
        log.info("RUN   %-34s (forced)", key)
    t0 = time.time()
    try:
        outputs, metrics = fn()
    except BaseException as exc:
        manifest.mark_failed(key, repr(exc))
        raise
    manifest.record(
        key, outputs, config_hash, input_hash, metrics, seconds=time.time() - t0, root=root
    )
    return {**manifest.entries[key], "skipped": False}
