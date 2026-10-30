"""Time LightGBM and XGBoost training on synthetic data to compare builds with and without OpenMP."""

from __future__ import annotations

import os
import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb


def main() -> None:
    """Train each booster once on a fixed dataset and print the wall time."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300_000, 60)).astype("float32")
    y = x[:, :5].sum(axis=1) + rng.normal(size=300_000).astype("float32")
    print(f"cpu_count={os.cpu_count()}", flush=True)

    # Time LightGBM including Dataset construction
    start = time.perf_counter()
    lgb.train({"verbose": -1}, lgb.Dataset(x, y), num_boost_round=100)
    print(f"lightgbm_seconds={time.perf_counter() - start:.2f}", flush=True)

    # Time XGBoost including DMatrix construction
    start = time.perf_counter()
    xgb.train({"tree_method": "hist", "max_depth": 6}, xgb.DMatrix(x, y), num_boost_round=100)
    print(f"xgboost_seconds={time.perf_counter() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
