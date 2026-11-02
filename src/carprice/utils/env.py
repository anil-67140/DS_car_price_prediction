"""Environment helpers: seeding, software-version capture, Colab detection, logging."""

from __future__ import annotations

import importlib.metadata as md
import logging
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np

TRACKED_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm", "catboost", "shap",
    "optuna", "mlflow", "pandera", "interpret", "joblib", "fastapi", "gradio", "pydantic",
    "evidently",
]


def in_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def software_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return out


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
