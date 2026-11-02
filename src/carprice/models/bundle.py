"""Versioned model bundle: save / load / compatibility validation.

A bundle directory contains EVERYTHING needed to serve a prediction from raw listing fields:

    model.joblib        full pipeline  FeatureBuilder -> ColumnTransformer -> estimator (+ log target)
    interval.joblib     fitted conformal calibrator
    metadata.json       version, data + config hashes, features, metrics, supported ranges, versions
    golden.json         a few raw inputs with the exact predictions made at training time
    MODEL_CARD.md       human-readable summary (written by the pipeline)

``load_bundle`` refuses to return a model unless the file hash matches the metadata, the
scikit-learn / numpy majors+minors match what trained it (pickles are version-fragile), and the
golden predictions are reproduced - that is the "model-loading and compatibility validation".
"""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib.metadata as md
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.env import software_versions
from ..utils.io import dump_joblib, load_joblib, read_json, sha256_file, write_json, write_text

log = logging.getLogger("carprice.bundle")

REQUIRED_FILES = ["model.joblib", "interval.joblib", "metadata.json", "golden.json"]
COMPAT_PACKAGES = ["scikit-learn", "numpy", "pandas"]  # major.minor must match


class BundleError(RuntimeError):
    pass


@dataclass
class Bundle:
    path: Path
    model: Any
    interval: Any
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return self.metadata["version"]


def make_version(project_version: str, model_hash: str) -> str:
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    return f"{project_version}+{ts}.{model_hash[:8]}"


def save_bundle(
    out_root: Path,
    model: Any,
    interval: Any,
    metadata: dict[str, Any],
    golden_inputs: pd.DataFrame,
    model_card: str,
    background: pd.DataFrame | None = None,
    dashboard: dict[str, Any] | None = None,
) -> Path:
    """Write a bundle atomically (file by file, metadata last = commit marker)."""
    tmp_model = out_root / "_staging"
    tmp_model.mkdir(parents=True, exist_ok=True)
    model_path = dump_joblib(tmp_model / "model.joblib", model)
    model_hash = sha256_file(model_path)
    version = make_version(metadata["project_version"], model_hash)
    bdir = out_root / version.replace("+", "_")
    bdir.mkdir(parents=True, exist_ok=True)
    final_model = bdir / "model.joblib"
    model_path.replace(final_model)
    dump_joblib(bdir / "interval.joblib", interval)
    golden_pred = np.asarray(model.predict(golden_inputs), dtype=float)
    write_json(bdir / "golden.json", {
        "inputs": golden_inputs.to_dict(orient="records"),
        "predictions": golden_pred.tolist(),
    })
    write_text(bdir / "MODEL_CARD.md", model_card)
    if background is not None:  # small raw-input sample used only as the SHAP reference distribution
        dump_joblib(bdir / "background.joblib", background)
    if dashboard is not None:
        write_json(bdir / "dashboard.json", dashboard)
    metadata = {
        **metadata,
        "version": version,
        "model_sha256": model_hash,
        "interval_sha256": sha256_file(bdir / "interval.joblib"),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "software_versions": software_versions(),
        "bundle_dir": bdir.name,
    }
    write_json(bdir / "metadata.json", metadata)  # written last: its presence marks a complete bundle
    with contextlib.suppress(OSError):
        tmp_model.rmdir()
    return bdir


def _mm(v: str) -> str:
    return ".".join(v.split(".")[:2])


def validate_environment(metadata: dict[str, Any], strict: bool = True) -> list[str]:
    """Return list of compatibility problems (empty = OK)."""
    problems = []
    trained = metadata.get("software_versions", {})
    for pkg in COMPAT_PACKAGES:
        try:
            cur = md.version(pkg)
        except md.PackageNotFoundError:
            problems.append(f"{pkg} not installed")
            continue
        if pkg in trained and _mm(trained[pkg]) != _mm(cur):
            problems.append(f"{pkg}: trained with {trained[pkg]}, running {cur}")
    if problems and strict:
        log.warning("Environment differs from training environment: %s", problems)
    return problems


def load_bundle(path: str | Path, strict_env: bool = False, check_golden: bool = True) -> Bundle:
    path = Path(path)
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise BundleError(f"Incomplete bundle {path.name}: missing {missing}")
    meta = read_json(path / "metadata.json")
    if sha256_file(path / "model.joblib") != meta["model_sha256"]:
        raise BundleError("model.joblib hash does not match metadata (file changed or corrupt)")
    if sha256_file(path / "interval.joblib") != meta["interval_sha256"]:
        raise BundleError("interval.joblib hash does not match metadata")
    problems = validate_environment(meta)
    if problems and strict_env:
        raise BundleError("Incompatible environment: " + "; ".join(problems))
    try:
        model = load_joblib(path / "model.joblib")
        interval = load_joblib(path / "interval.joblib")
    except Exception as exc:
        raise BundleError(f"Could not unpickle model (likely library version mismatch): {exc}") from exc
    if check_golden:
        g = read_json(path / "golden.json")
        got = np.asarray(model.predict(pd.DataFrame(g["inputs"])), dtype=float)
        exp = np.asarray(g["predictions"], dtype=float)
        if not np.allclose(got, exp, rtol=1e-6):
            raise BundleError(
                f"Golden predictions not reproduced (max rel diff {np.max(np.abs(got - exp) / exp):.2e}). "
                f"Environment problems: {problems or 'none detected'}"
            )
    return Bundle(path=path, model=model, interval=interval, metadata=meta)


def latest_bundle_dir(models_root: Path) -> Path:
    """Champion pointer if present, else newest complete bundle."""
    reg = models_root / "registry.json"
    if reg.exists():
        champ = read_json(reg).get("champion")
        if champ and (models_root / champ / "metadata.json").exists():
            return models_root / champ
    cands = sorted(p for p in models_root.glob("*") if (p / "metadata.json").exists())
    if not cands:
        raise BundleError(f"No model bundle found under {models_root}. Run the pipeline first.")
    return cands[-1]


def register(models_root: Path, bundle_dir: Path, promote: bool, note: str = "") -> dict[str, Any]:
    reg_path = models_root / "registry.json"
    reg = read_json(reg_path) if reg_path.exists() else {"champion": None, "versions": []}
    meta = read_json(bundle_dir / "metadata.json")
    reg["versions"] = [v for v in reg["versions"] if v["dir"] != bundle_dir.name]
    reg["versions"].append({"dir": bundle_dir.name, "version": meta["version"],
                            "created_utc": meta["created_utc"], "note": note,
                            "test_mae_inr": meta.get("test_metrics", {}).get("mae_inr")})
    if promote or reg["champion"] is None:
        reg["champion"] = bundle_dir.name
    write_json(reg_path, reg)
    return reg
