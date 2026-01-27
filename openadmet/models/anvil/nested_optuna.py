"""
Minimal nested-CV runner using OptunaSearchCV as inner search.

This module exposes `NestedSearchConfig` and `run_nested_optuna_search`.
It is intentionally light-weight so it can be imported by existing Anvil
trainers or executed from a thin wrapper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from optuna import create_study
from optuna.integration import OptunaSearchCV  # type: ignore
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.utils.multiclass import type_of_target

logger = logging.getLogger(__name__)


@dataclass
class NestedSearchConfig:
    """Config for nested Optuna search."""

    outer_n_splits: int = 5
    outer_repeats: int = 1
    outer_shuffle: bool = True
    outer_random_state: int | None = 42

    inner_cv: int = 3
    n_trials: int = 50
    timeout_per_trial_s: int | None = None
    sampler_seed: int | None = None

    scoring: str | dict | None = None  # Metrics for outer CV evaluation
    hpo_scoring: str | None = None  # Metric for inner HPO optimization
    n_jobs_outer: int = 1  # for cross_validate outer loop

    # Custom CV splitter (e.g., for scaffold/cluster-based splits)
    custom_outer_cv: Any | None = None


def _make_outer_cv(cfg: NestedSearchConfig, y: np.ndarray):
    """
    Create outer CV splitter based on target type or custom splitter.

    Args:
        cfg: NestedSearchConfig instance.
        y: Target array to determine if task is classification or regression.

    Returns:
        CV splitter (custom, stratified for classification, or regular for
        regression).

    """
    # Use custom splitter if provided (e.g., scaffold/cluster-based)
    if cfg.custom_outer_cv is not None:
        logger.debug(
            f"Using custom outer CV splitter: {cfg.custom_outer_cv.__class__.__name__}"
        )
        return cfg.custom_outer_cv

    # Fall back to default sklearn splitters
    target_type = type_of_target(y)
    is_classification = target_type in ("binary", "multiclass")

    if cfg.outer_repeats and cfg.outer_repeats > 1:
        if is_classification:
            return RepeatedStratifiedKFold(
                n_splits=cfg.outer_n_splits,
                n_repeats=cfg.outer_repeats,
                random_state=cfg.outer_random_state,
            )
        else:
            return RepeatedKFold(
                n_splits=cfg.outer_n_splits,
                n_repeats=cfg.outer_repeats,
                random_state=cfg.outer_random_state,
            )
    else:
        if is_classification:
            return StratifiedKFold(
                n_splits=cfg.outer_n_splits,
                shuffle=cfg.outer_shuffle,
                random_state=cfg.outer_random_state,
            )
        else:
            return KFold(
                n_splits=cfg.outer_n_splits,
                shuffle=cfg.outer_shuffle,
                random_state=cfg.outer_random_state,
            )


def _make_optuna_search(
    base_estimator: BaseEstimator,
    param_distributions: dict[str, Any],
    cfg: NestedSearchConfig,
) -> OptunaSearchCV:
    """
    Construct unfitted OptunaSearchCV object with TPESampler-backed Study.

    Note: `param_distributions` must use `optuna.distributions.*` objects.

    Args:
        base_estimator: sklearn estimator or Pipeline (unfitted).
        param_distributions: optuna.distributions for hyperparams.
        cfg: NestedSearchConfig instance.

    Returns:
        OptunaSearchCV object.

    """
    sampler = (
        TPESampler(seed=cfg.sampler_seed)
        if cfg.sampler_seed is not None
        else TPESampler()
    )
    study = create_study(sampler=sampler, direction="maximize")

    # OptunaSearchCV requires a single scoring metric for HPO
    # Use hpo_scoring if provided, otherwise extract from scoring dict
    if cfg.hpo_scoring:
        # If hpo_scoring is provided and scoring is a dict, look up the scorer
        if isinstance(cfg.scoring, dict):
            # Try to get the scorer object from the dict, fall back to string
            # This handles custom metrics like 'spearmanr' and 'ktau'
            hpo_scoring = cfg.scoring.get(cfg.hpo_scoring, cfg.hpo_scoring)
        else:
            hpo_scoring = cfg.hpo_scoring
    elif isinstance(cfg.scoring, dict):
        # Fall back to first metric if hpo_scoring not specified
        hpo_scoring = list(cfg.scoring.values())[0]
    else:
        hpo_scoring = cfg.scoring

    search = OptunaSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_trials=cfg.n_trials,
        cv=cfg.inner_cv,
        scoring=hpo_scoring,
        study=study,
        n_jobs=1,  # Always use 1 to avoid nested parallelism with outer CV
        verbose=0,
        return_train_score=False,
    )
    return search


def run_nested_optuna_search(
    X: np.ndarray,
    y: np.ndarray,
    base_estimator: BaseEstimator,
    param_distributions: dict[str, Any],
    cfg: NestedSearchConfig,
) -> dict[str, Any]:
    """
    Run nested CV: outer CV where inner search = OptunaSearchCV.

    Args:
        X: feature matrix (n_samples, n_features).
        y: label vector (n_samples,).
        base_estimator: sklearn estimator or Pipeline (unfitted).
        param_distributions: optuna.distributions for hyperparams (keys use
            sklearn param names).
        cfg: NestedSearchConfig instance.

    Returns:
        dict with:
            - 'outer_cv_results': raw sklearn.cross_validate return dict
            - 'outer_best_params': list of best_params_ from each fitted
                OptunaSearchCV
            - 'outer_best_scores': list of best_score_ from each fitted
                OptunaSearchCV
            - 'estimators': list of fitted OptunaSearchCV objects (one per
                outer fold)

    """
    outer_cv = _make_outer_cv(cfg, y)
    optuna_search = _make_optuna_search(base_estimator, param_distributions, cfg)

    logger.info(
        "Starting nested CV: outer=%s inner_cv=%d n_trials=%d",
        outer_cv,
        cfg.inner_cv,
        cfg.n_trials,
    )

    cv_results = cross_validate(
        optuna_search,
        X,
        y,
        cv=outer_cv,
        scoring=cfg.scoring,
        return_estimator=True,
        n_jobs=cfg.n_jobs_outer,
        verbose=0,
    )

    fitted_searches: list[OptunaSearchCV] = [est for est in cv_results["estimator"]]
    outer_best_params = [getattr(s, "best_params_", None) for s in fitted_searches]
    outer_best_scores = [getattr(s, "best_score_", None) for s in fitted_searches]

    return {
        "outer_cv_results": cv_results,
        "outer_best_params": outer_best_params,
        "outer_best_scores": outer_best_scores,
        "estimators": fitted_searches,
    }
