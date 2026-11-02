"""Hyper-parameter optimisation: GridSearchCV (small spaces), RandomizedSearchCV (forests),
Optuna/TPE (boosting, SVR). All searches use TRAINING data only, with grouped folds that are
different from the comparison folds. The test set is never seen here.

Optuna studies live in an append-only journal file so an interrupted Colab run continues from the
completed trials instead of starting over.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .train import cv_experiment
from .zoo import build_model

log = logging.getLogger("carprice.tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)

GRID_MODELS = {
    "ridge": {"regressor__model__alpha": [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
    "lasso": {"regressor__model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]},
    "elasticnet": {"regressor__model__alpha": [1e-4, 1e-3, 3e-3, 1e-2, 3e-2],
                   "regressor__model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
}
RANDOM_MODELS = {
    "random_forest": {
        "regressor__model__n_estimators": [200, 400, 600],
        "regressor__model__max_depth": [None, 6, 10, 16],
        "regressor__model__min_samples_leaf": [1, 2, 3, 5],
        "regressor__model__max_features": [0.3, 0.5, 0.7, 1.0],
    },
    "extra_trees": {
        "regressor__model__n_estimators": [200, 400, 600],
        "regressor__model__max_depth": [None, 8, 12, 20],
        "regressor__model__min_samples_leaf": [1, 2, 3, 5],
        "regressor__model__max_features": [0.4, 0.6, 0.8, 1.0],
    },
}
OPTUNA_MODELS = ["hist_gradient_boosting", "lightgbm", "xgboost", "catboost", "gradient_boosting", "svr"]
TUNABLE = [*GRID_MODELS, *RANDOM_MODELS, *OPTUNA_MODELS]


def method_for(name: str) -> str | None:
    if name in GRID_MODELS:
        return "grid"
    if name in RANDOM_MODELS:
        return "random"
    if name in OPTUNA_MODELS:
        return "optuna"
    return None


def _suggest(trial: optuna.Trial, name: str) -> dict[str, Any]:
    f = trial.suggest_float
    i = trial.suggest_int
    if name == "hist_gradient_boosting":
        return dict(max_iter=i("max_iter", 100, 500), learning_rate=f("learning_rate", 0.02, 0.2, log=True),
                    max_leaf_nodes=i("max_leaf_nodes", 4, 31), min_samples_leaf=i("min_samples_leaf", 3, 30),
                    l2_regularization=f("l2_regularization", 1e-3, 10, log=True))
    if name == "lightgbm":
        return dict(n_estimators=i("n_estimators", 100, 600), learning_rate=f("learning_rate", 0.02, 0.2, log=True),
                    num_leaves=i("num_leaves", 4, 31), min_child_samples=i("min_child_samples", 3, 30),
                    subsample=f("subsample", 0.5, 1.0), colsample_bytree=f("colsample_bytree", 0.4, 1.0),
                    reg_lambda=f("reg_lambda", 1e-3, 10, log=True))
    if name == "xgboost":
        return dict(n_estimators=i("n_estimators", 100, 600), learning_rate=f("learning_rate", 0.02, 0.2, log=True),
                    max_depth=i("max_depth", 2, 7), min_child_weight=i("min_child_weight", 1, 10),
                    subsample=f("subsample", 0.5, 1.0), colsample_bytree=f("colsample_bytree", 0.4, 1.0),
                    reg_lambda=f("reg_lambda", 1e-3, 10, log=True))
    if name == "catboost":
        return dict(iterations=i("iterations", 200, 700), depth=i("depth", 3, 8),
                    learning_rate=f("learning_rate", 0.02, 0.2, log=True), l2_leaf_reg=f("l2_leaf_reg", 1, 10))
    if name == "gradient_boosting":
        return dict(n_estimators=i("n_estimators", 100, 500), learning_rate=f("learning_rate", 0.02, 0.2, log=True),
                    max_depth=i("max_depth", 2, 5), subsample=f("subsample", 0.5, 1.0),
                    min_samples_leaf=i("min_samples_leaf", 1, 10))
    if name == "svr":
        return dict(C=f("C", 0.3, 100, log=True), epsilon=f("epsilon", 0.005, 0.3, log=True),
                    gamma=f("gamma", 1e-3, 1.0, log=True))
    raise ValueError(name)


def _strip(params: dict[str, Any]) -> dict[str, Any]:
    return {k.split("__")[-1]: v for k, v in params.items()}


def tune_model(
    name: str,
    variant: str,
    X: pd.DataFrame,
    y: pd.Series,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    build_kwargs: dict[str, Any],
    *,
    optuna_trials: int,
    random_iter: int,
    seed: int,
    journal_dir: Path,
    timeout: float | None = None,
) -> dict[str, Any]:
    method = method_for(name)
    if method is None:
        raise ValueError(f"No search space defined for {name}")
    t0 = time.perf_counter()
    base = build_model(name, variant, **build_kwargs)
    result: dict[str, Any] = {"model": name, "variant": variant, "method": method}

    if method in ("grid", "random"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "grid":
                search = GridSearchCV(base, GRID_MODELS[name], cv=inner_splits,
                                      scoring="neg_mean_absolute_error", n_jobs=1, refit=False)
            else:
                search = RandomizedSearchCV(base, RANDOM_MODELS[name], n_iter=random_iter, cv=inner_splits,
                                            scoring="neg_mean_absolute_error", n_jobs=1, refit=False,
                                            random_state=seed)
            search.fit(X, y)
        best = _strip(search.best_params_)
        result.update(best_params=best, best_inner_mae=float(-search.best_score_),
                      n_trials=len(search.cv_results_["params"]))
    else:
        journal_dir.mkdir(parents=True, exist_ok=True)
        storage = JournalStorage(JournalFileBackend(str(journal_dir / f"{name}_{variant}.journal")))
        study = optuna.create_study(
            study_name=f"{name}_{variant}", storage=storage, load_if_exists=True, direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        )
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        def objective(trial: optuna.Trial) -> float:
            params = _suggest(trial, name)
            r = cv_experiment(name, variant, X, y, inner_splits, build_kwargs, n_repeats=1,
                              params=params, measure_size=False)
            if r["status"] != "ok":
                raise optuna.TrialPruned()
            trial.set_user_attr("mae_se", r["cv"]["mae_inr_se"])
            return r["cv"]["mae_inr_mean"]

        remaining = max(optuna_trials - done, 0)
        if remaining:
            study.optimize(objective, n_trials=remaining, timeout=timeout, gc_after_trial=True)
        result.update(best_params=dict(study.best_params), best_inner_mae=float(study.best_value),
                      n_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                      resumed_trials=done)
    result["search_seconds"] = round(time.perf_counter() - t0, 1)
    return result
