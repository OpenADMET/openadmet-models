import numpy as np
import pytest
from numpy.testing import assert_allclose

from openadmet.models.active_learning.committee import (
    _QUERY_STRATEGIES,
    CommitteeRegressor,
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
        model = LGBMRegressorModel(**mod_params)

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
    committee = CommitteeRegressor.from_models(models=models)

    # Query
    y_query = committee.query(X, query_strategy=query_strategy)

    # Predict
    y_pred, y_pred_std = committee.predict(X, return_std=True)


def test_save_load(tmp_path, models, eval_data):
    # Model parameters
    mod_params = {
        "n_estimators": 5,
        "force_row_wise": True,
    }

    # Data
    X, y = eval_data

    # Create committee
    committee = CommitteeRegressor.from_models(models=models)

    # Predict before saving
    preds = committee.predict(X)

    # Save and load
    save_paths = [tmp_path / "committee_model_{i}.pkl" for i in range(len(models))]
    committee.save(save_paths)
    committee.load(
        save_paths,
        models=[LGBMRegressorModel(**mod_params) for _ in save_paths],
    )

    # Predict after loading
    preds2 = committee.predict(X)

    # Check that predictions are the same
    assert_allclose(preds, preds2)


def test_serialization(tmp_path, models, eval_data):
    # Data
    X, y = eval_data

    # Create committee
    committee = CommitteeRegressor.from_models(models=models)

    # Predict before saving
    preds = committee.predict(X)

    # Save and load
    param_paths = [tmp_path / "committee_model_{i}.json" for i in range(len(models))]
    serial_paths = [tmp_path / "committee_model_{i}.pkl" for i in range(len(models))]
    committee.serialize(param_paths, serial_paths)
    committee.deserialize(param_paths, serial_paths, mod_class=models[0].__class__)

    # Predict after loading
    preds2 = committee.predict(X)

    # Check that predictions are the same
    assert_allclose(preds, preds2)
