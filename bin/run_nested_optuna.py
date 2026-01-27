#!/usr/bin/env python3
"""
CLI runner for nested Optuna Search.

Usage:
  - prepare X (features) and y (labels) as numpy .npy files
    (X: (n, d), y: (n,))
  - write a small python file `params.py` that defines
    `param_distributions` using optuna.distributions (example below).
  - run:
      ./bin/run_nested_optuna.py --X X.npy --y y.npy \
        --params-file params.py --out results.json

Example params.py:
    from optuna.distributions import (
        FloatDistribution,
        IntDistribution,
        CategoricalDistribution
    )

    param_distributions = {
        "clf__C": FloatDistribution(1e-4, 1e2, log=True),
        "clf__penalty": CategoricalDistribution(["l2"]),
    }
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from openadmet.models.anvil.nested_optuna import (
    NestedSearchConfig,
    run_nested_optuna_search,
)

logger = logging.getLogger(__name__)


def load_param_module(path: str) -> dict[str, Any]:
    """
    Load param_distributions from a Python file.

    Args:
        path: Path to Python file defining param_distributions.

    Returns:
        Dictionary of parameter distributions.

    """
    spec = importlib.util.spec_from_file_location("user_params", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "param_distributions"):
        raise RuntimeError("params file must define `param_distributions` dict")
    return getattr(mod, "param_distributions")


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.

    """
    p = argparse.ArgumentParser(
        description="Run nested OptunaSearchCV (outer CV + inner Optuna)"
    )
    p.add_argument("--X", required=True, help="path to X.npy (features)")
    p.add_argument("--y", required=True, help="path to y.npy (labels)")
    p.add_argument(
        "--params-file",
        required=True,
        help="python file that defines param_distributions",
    )
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--inner-cv", type=int, default=3)
    p.add_argument("--outer-n-splits", type=int, default=5)
    p.add_argument("--outer-repeats", type=int, default=1)
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--out", default="nested_optuna_results.json")
    p.add_argument("--n-jobs-outer", type=int, default=1)
    p.add_argument("--optuna-n-jobs", type=int, default=1)
    p.add_argument("--scoring", default=None)
    return p.parse_args()


def main():
    """Run nested Optuna search from command line."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    X = np.load(args.X)
    y = np.load(args.y)

    param_distributions = load_param_module(os.path.abspath(args.params_file))  # noqa: E501

    # simple default pipeline: scaler + logistic classifier
    # (users usually provide Pipeline as base_estimator)
    base_estimator = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]
    )

    cfg = NestedSearchConfig(
        outer_n_splits=args.outer_n_splits,
        outer_repeats=args.outer_repeats,
        outer_random_state=args.sampler_seed,
        inner_cv=args.inner_cv,
        n_trials=args.n_trials,
        sampler_seed=args.sampler_seed,
        optuna_n_jobs=args.optuna_n_jobs,
        scoring=args.scoring,
        n_jobs_outer=args.n_jobs_outer,
    )

    results = run_nested_optuna_search(X, y, base_estimator, param_distributions, cfg)

    # summarize outer results (best params and best scores)
    summary = {
        "n_outer_folds": len(results["outer_best_params"]),
        "outer_best_scores": results["outer_best_scores"],
        "outer_best_params": results["outer_best_params"],
    }

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Wrote results summary to %s", args.out)


if __name__ == "__main__":
    main()
