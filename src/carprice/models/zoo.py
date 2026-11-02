"""Model registry: every candidate regressor behind one factory, same preprocessing, same splits.

All models except the two raw-scale baselines are fit on ``log1p(price)`` through
``TransformedTargetRegressor`` and evaluated back on the INR scale. Reasoning: used-car prices are
right-skewed and errors are multiplicative (a 10% miss matters equally on cheap and dear cars), so
log-target training stabilises variance and keeps predictions positive.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..features.engineering import FeatureBuilder, build_preprocessor


@dataclass
class ModelSpec:
    name: str
    family: str                         # baseline | linear | tree | boosting | kernel | neighbors | neural | glassbox
    factory: Callable[[int, int], Any]  # (seed, n_jobs) -> estimator
    scaler: str = "robust"              # robust | standard | none
    log_target: bool = True
    requires: str | None = None         # import name needed
    notes: str = ""
    default_params: dict[str, Any] = field(default_factory=dict)

    def available(self) -> tuple[bool, str]:
        if self.requires and importlib.util.find_spec(self.requires) is None:
            return False, f"package '{self.requires}' not installed"
        return True, ""


def _xgb(seed, n_jobs, **kw):
    from xgboost import XGBRegressor

    p = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0)
    p.update(kw)
    return XGBRegressor(random_state=seed, n_jobs=n_jobs, verbosity=0, **p)


def _lgbm(seed, n_jobs, **kw):
    from lightgbm import LGBMRegressor

    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=8,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
    p.update(kw)
    return LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1, **p)


def _cat(seed, n_jobs, **kw):
    from catboost import CatBoostRegressor

    p = dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0)
    p.update(kw)
    return CatBoostRegressor(random_seed=seed, thread_count=n_jobs, verbose=0,
                             allow_writing_files=False, **p)


def _ebm(seed, n_jobs, **kw):
    from interpret.glassbox import ExplainableBoostingRegressor

    p = dict(interactions=5, max_rounds=400, learning_rate=0.03)
    p.update(kw)
    return ExplainableBoostingRegressor(random_state=seed, n_jobs=n_jobs, **p)


def _rf(seed, n_jobs, **kw):
    p = dict(n_estimators=300, min_samples_leaf=2, max_features=0.6)
    p.update(kw)
    return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **p)


def _et(seed, n_jobs, **kw):
    p = dict(n_estimators=300, min_samples_leaf=2, max_features=0.8)
    p.update(kw)
    return ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **p)


def _gbr(seed, n_jobs, **kw):
    p = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, min_samples_leaf=3)
    p.update(kw)
    return GradientBoostingRegressor(random_state=seed, **p)


def _hgb(seed, n_jobs, **kw):
    p = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=1.0)
    p.update(kw)
    return HistGradientBoostingRegressor(random_state=seed, **p)


SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("mean_baseline", "baseline", lambda s, n: DummyRegressor(strategy="mean"),
              scaler="none", log_target=False,
              notes="Predicts the arithmetic mean training price. The floor every model must beat."),
    ModelSpec("median_baseline", "baseline", lambda s, n: DummyRegressor(strategy="median"),
              scaler="none", log_target=False,
              notes="Predicts the median training price (robust to the expensive tail). Also the MASE reference."),
    ModelSpec("linear", "linear", lambda s, n: LinearRegression(),
              notes="OLS on log price. Interpretable; unstable when many one-hot columns are collinear."),
    ModelSpec("ridge", "linear", lambda s, n, **k: Ridge(alpha=k.get("alpha", 3.0), random_state=s)),
    ModelSpec("lasso", "linear", lambda s, n, **k: Lasso(alpha=k.get("alpha", 0.003), max_iter=50000, random_state=s),
              notes="L1: sparse coefficients, doubles as embedded feature selection."),
    ModelSpec("elasticnet", "linear",
              lambda s, n, **k: ElasticNet(alpha=k.get("alpha", 0.004), l1_ratio=k.get("l1_ratio", 0.5),
                                           max_iter=50000, random_state=s)),
    ModelSpec("huber", "linear", lambda s, n: HuberRegressor(alpha=1e-2, epsilon=1.35, max_iter=3000),
              notes="Outlier-robust linear fit - relevant for mis-priced listings."),
    ModelSpec("knn", "neighbors", lambda s, n: KNeighborsRegressor(n_neighbors=7, weights="distance"),
              notes="Distance-based; sensitive to feature scaling and one-hot dimensionality."),
    ModelSpec("decision_tree", "tree", lambda s, n: DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, random_state=s),
              scaler="none"),
    ModelSpec("random_forest", "tree", _rf, scaler="none"),
    ModelSpec("extra_trees", "tree", _et, scaler="none"),
    ModelSpec("gradient_boosting", "boosting", _gbr, scaler="none"),
    ModelSpec("hist_gradient_boosting", "boosting", _hgb, scaler="none"),
    ModelSpec("xgboost", "boosting", _xgb, scaler="none", requires="xgboost"),
    ModelSpec("lightgbm", "boosting", _lgbm, scaler="none", requires="lightgbm"),
    ModelSpec("catboost", "boosting", _cat, scaler="none", requires="catboost",
              notes="Run on the shared one-hot features (not CatBoost's native categorical handling) "
                    "for a like-for-like comparison."),
    ModelSpec("svr", "kernel", lambda s, n, **k: SVR(C=k.get("C", 3.0), epsilon=k.get("epsilon", 0.05),
                                                    gamma=k.get("gamma", "scale")),
              notes="RBF kernel; needs scaled inputs."),
    ModelSpec("ebm", "glassbox", _ebm, scaler="none", requires="interpret",
              notes="Explainable Boosting Machine: additive + a few pairwise terms, intrinsically interpretable."),
    ModelSpec("mlp", "neural",
              lambda s, n: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                        validation_fraction=0.15, max_iter=3000, random_state=s),
              scaler="standard",  # robust scaling let heavy-tailed km/year inputs blow up the net
              notes="Included for completeness. With <1k tabular rows a neural net is not expected to "
                    "win; if it does not, that is the (documented) result."),
]}

SPECS["ridge"].default_params = {"alpha": 3.0}


def available_models(include: list[str]) -> tuple[list[str], dict[str, str]]:
    ok, excluded = [], {}
    for name in include:
        if name not in SPECS:
            excluded[name] = "unknown model name in config"
            continue
        good, why = SPECS[name].available()
        if good:
            ok.append(name)
        else:
            excluded[name] = why
    return ok, excluded


def build_model(
    name: str,
    variant: str,
    reference_year: int,
    available_optional: list[str] | None = None,
    seed: int = 42,
    n_jobs: int = 2,
    n_model_tokens: int = 2,
    min_frequency: int = 3,
    params: dict[str, Any] | None = None,
    scaler: str | None = None,
):
    """Full raw-input -> INR-price estimator (FeatureBuilder -> ColumnTransformer -> model)."""
    spec = SPECS[name]
    est = spec.factory(seed, n_jobs, **(params or {})) if params else spec.factory(seed, n_jobs)
    # scale-sensitive spec decides; trees skip scaling (no effect, saves time)
    pipe = Pipeline([
        ("features", FeatureBuilder(reference_year=reference_year, n_model_tokens=n_model_tokens)),
        ("prep", build_preprocessor(variant, available_optional, scaler or spec.scaler,
                                    min_frequency=min_frequency, seed=seed)),
        ("model", est),
    ])
    if spec.log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1,
                                          check_inverse=False)
    return pipe
