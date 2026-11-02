"""MLflow tracking wrapper. Tracking must never be able to break training, so every call is guarded."""

from __future__ import annotations

import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

log = logging.getLogger("carprice.tracking")


def git_sha(cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:
        return None


class Tracker:
    def __init__(self, tracking_dir: Path, experiment: str, enabled: bool = True):
        self.enabled = enabled
        self.mlflow = None
        if not enabled:
            return
        try:
            import mlflow

            tracking_dir.mkdir(parents=True, exist_ok=True)
            # SQLite backend (the file store is in maintenance mode in MLflow 3.x)
            mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}")
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=(tracking_dir / "artifacts").resolve().as_uri())
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow
        except Exception as exc:
            log.warning("MLflow disabled: %s", exc)
            self.enabled = False

    @contextmanager
    def run(self, name: str, tags: dict[str, Any] | None = None, nested: bool = False):
        if not self.enabled:
            yield None
            return
        try:
            ctx = self.mlflow.start_run(run_name=name, nested=nested, tags={k: str(v) for k, v in (tags or {}).items()})
        except Exception as exc:
            log.warning("MLflow start_run failed: %s", exc)
            yield None
            return
        with ctx as r:
            yield r

    def log(self, params: dict[str, Any] | None = None, metrics: dict[str, float] | None = None,
            artifacts: list[Path] | None = None) -> None:
        if not self.enabled:
            return
        try:
            if params:
                self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            if metrics:
                self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                         if v is not None and v == v and abs(float(v)) != float("inf")})
            for a in artifacts or []:
                if Path(a).exists():
                    self.mlflow.log_artifact(str(a))
        except Exception as exc:
            log.warning("MLflow logging failed: %s", exc)

    def log_model(self, model, name: str = "model") -> None:
        if not self.enabled:
            return
        try:
            self.mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
        except Exception as exc:
            log.warning("MLflow log_model failed: %s", exc)
