"""Configuration loading and project paths. No hard-coded paths or credentials anywhere."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .utils.io import stable_hash

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # repo root in a source checkout
CONFIG_DIR_ENV = "CARPRICE_CONFIG_DIR"
HOME_ENV = "CARPRICE_HOME"


def _config_dir() -> Path:
    env = os.environ.get(CONFIG_DIR_ENV)
    return Path(env) if env else PACKAGE_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


# Small, fast configuration used by smoke tests, CI and `--quick` runs (NOT for reported results).
QUICK_OVERRIDES: dict[str, Any] = {
    "models": {"include": ["median_baseline", "ridge", "lasso", "hist_gradient_boosting", "xgboost"], "n_jobs": 2},
    "tuning": {"top_k": 2, "optuna_trials": 6, "random_search_iter": 6, "timeout_seconds_per_model": 120},
    "split": {"cv_repeats": 1},
    "explainability": {"shap_sample": 60},
}


@dataclass
class Paths:
    home: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    artifacts: Path
    reports: Path
    checkpoints: Path
    logs: Path

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def metrics(self) -> Path:
        return self.reports / "metrics"

    @property
    def model_reports(self) -> Path:
        return self.reports / "model_reports"

    @property
    def manifest(self) -> Path:
        return self.checkpoints / "manifest.json"

    @property
    def models(self) -> Path:
        return self.artifacts / "models"

    def make_dirs(self) -> None:
        for p in (
            self.data_raw, self.data_interim, self.data_processed, self.artifacts,
            self.figures, self.metrics, self.model_reports, self.checkpoints, self.logs,
            self.models,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    cfg: dict[str, Any]
    datasets: dict[str, Any]
    paths: Paths

    # convenience ---------------------------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.cfg["project"]["seed"])

    @property
    def dataset_key(self) -> str:
        return self.cfg["dataset"]["active"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.datasets[self.dataset_key]

    def section_hash(self, *sections: str) -> str:
        return stable_hash({s: self.cfg.get(s) for s in sections} | {"dataset": self.dataset_key})

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_settings(
    overrides: dict[str, Any] | None = None, home: str | Path | None = None,
    dataset_overrides: dict[str, Any] | None = None,
) -> Settings:
    cdir = _config_dir()
    cfg = load_yaml(cdir / "config.yaml")
    if overrides:
        cfg = deep_update(cfg, overrides)
    datasets = load_yaml(cdir / "datasets.yaml")
    if dataset_overrides:
        datasets = {**datasets, **dataset_overrides}

    home_path = home or cfg["paths"].get("home") or os.environ.get(HOME_ENV) or PACKAGE_ROOT
    home_path = Path(home_path).expanduser().resolve()
    p = cfg["paths"]
    paths = Paths(
        home=home_path,
        data_raw=home_path / p["data_raw"],
        data_interim=home_path / p["data_interim"],
        data_processed=home_path / p["data_processed"],
        artifacts=home_path / p["artifacts"],
        reports=home_path / p["reports"],
        checkpoints=home_path / p["checkpoints"],
        logs=home_path / p["logs"],
    )
    return Settings(cfg=cfg, datasets=datasets, paths=paths)
