import numpy as np
import pytest

from openadmet.models.active_learning.committee import (
    _QUERY_STRATEGIES,
    ActiveLearningCommitteeRegressor,
)
from openadmet.models.architecture.lgbm import LGBMRegressorModel


@pytest.fixture
def train_data():
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    return X, y


@pytest.fixture
def eval_data():
    np.random.seed(1234)
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 1)
    return X, y


@pytest.fixture
def models(train_data):
    # Data
    X, y = train_data

    # Model parameters
    mod_params = {
        "n_estimators": 5,
        "force_row_wise": True,
    }

    # Initialize set of models
    models = []
    for i in range(5):
        # Initialize model
        model = LGBMRegressorModel.from_params(mod_params=mod_params)

        # Train
        bootstrap_idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        model.train(X[bootstrap_idx, :], y[bootstrap_idx, :])

        # Add to list
        models.append(model)

    return models


@pytest.mark.parametrize("query_strategy", _QUERY_STRATEGIES.keys())
def test_committee(query_strategy, models, eval_data):
    # Data
    X, y = eval_data

    # Create committee
    alr = ActiveLearningCommitteeRegressor.from_models(
        models=models, query_strategy=query_strategy
    )

    # Query
    idx, instances = alr.query(X, n_instances=1)

    # Predict
    y_pred, y_pred_std = alr.predict(X, return_std=True)
