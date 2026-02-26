import numpy as np
import pytest
from numpy.testing import assert_allclose

from openadmet.models.active_learning.acquisition import _ACQUISITION_FUNCTIONS
from openadmet.models.active_learning.committee import CommitteeRegressor
from openadmet.models.architecture.dummy import DummyRegressorModel


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 3))
    y = (
        1.2 * X[:, [0]]
        - 0.8 * X[:, [1]]
        + 0.3 * X[:, [2]]
        + 0.1 * rng.normal(size=(120, 1))
    )
    return X[:80], X[80:100], X[100:], y[:80], y[80:100], y[100:]


@pytest.fixture
def dummy_models():
    models = []
    for _ in range(5):
        model = DummyRegressorModel(strategy="mean")
        models.append(model)
    return models


@pytest.fixture
def trained_committee(dummy_models, toy_data):
    X_train, X_val, _, y_train, y_val, _ = toy_data
    rng = np.random.default_rng(123)
    for model in dummy_models:
        bootstrap_idx = rng.choice(
            X_train.shape[0], size=X_train.shape[0], replace=True
        )
        model.train(X_train[bootstrap_idx], y_train[bootstrap_idx])
    return CommitteeRegressor.from_models(models=dummy_models), X_val, y_val


@pytest.mark.parametrize("query_strategy", sorted(_ACQUISITION_FUNCTIONS.keys()))
def test_committee_query_predict(trained_committee, query_strategy):
    committee, X_val, _ = trained_committee
    y_query = committee.query(X_val, query_strategy=query_strategy)
    y_pred, y_pred_std = committee.predict(X_val, return_std=True)
    assert y_query.shape == (X_val.shape[0], 1)
    assert y_pred.shape == (X_val.shape[0], 1)
    assert y_pred_std.shape == (X_val.shape[0], 1)
    assert np.isfinite(y_query).all()


def test_invalid_query_strategy_raises(trained_committee):
    committee, X_val, _ = trained_committee
    with pytest.raises(ValueError):
        committee.query(X_val, query_strategy="not-a-strategy")


def test_invalid_calibration_method_raises(trained_committee):
    committee, X_val, y_val = trained_committee
    with pytest.raises(ValueError):
        committee.calibrate_uncertainty(X_val, y_val, method="not-a-method")


@pytest.mark.parametrize(
    "calibration_method", ["isotonic-regression", "scaling-factor"]
)
def test_calibration_paths(trained_committee, calibration_method):
    committee, X_val, y_val = trained_committee
    committee.calibrate_uncertainty(X_val, y_val, method=calibration_method)
    assert committee.calibrated
    y_pred, y_std = committee.predict(X_val, return_std=True)
    assert y_pred.shape == y_std.shape == y_val.shape


def test_train_and_train_validation(toy_data):
    X_train, _, X_test, y_train, _, _ = toy_data
    committee = CommitteeRegressor.train(
        X_train, y_train, mod_class=DummyRegressorModel, n_models=4
    )
    mean, std = committee.predict(X_test, return_std=True)
    assert committee.n_models == 4
    assert mean.shape == std.shape == (X_test.shape[0], 1)
    with pytest.raises(ValueError):
        CommitteeRegressor.train(X_train, y_train, mod_class=None, n_models=2)


@pytest.mark.parametrize(
    "calibration_method", ["isotonic-regression", "scaling-factor", None]
)
def test_save_load_roundtrip(tmp_path, trained_committee, calibration_method):
    committee, X_val, y_val = trained_committee
    calibration_model_path = (
        tmp_path / "calibration_model.pkl" if calibration_method is not None else None
    )
    if calibration_method is not None:
        committee.calibrate_uncertainty(X_val, y_val, method=calibration_method)
    y_pred_mean, y_pred_std = committee.predict(X_val, return_std=True)
    save_paths = [
        tmp_path / f"committee_model_{i}.pkl" for i in range(committee.n_models)
    ]
    committee.save(save_paths, calibration_path=calibration_model_path)
    models_new = [model.make_new() for model in committee.models]
    [model.build() for model in models_new]
    committee = committee.load(
        save_paths, models=models_new, calibration_path=calibration_model_path
    )
    y_pred_mean2, y_pred_std2 = committee.predict(X_val, return_std=True)
    assert_allclose(y_pred_mean, y_pred_mean2)
    assert_allclose(y_pred_std, y_pred_std2)
    assert committee.calibrated is (calibration_method is not None)


@pytest.mark.parametrize(
    "calibration_method", ["isotonic-regression", "scaling-factor", None]
)
def test_serialize_deserialize_roundtrip(
    tmp_path, trained_committee, calibration_method
):
    committee, X_val, y_val = trained_committee
    calibration_model_path = (
        tmp_path / "calibration_model.pkl" if calibration_method is not None else None
    )
    if calibration_method is not None:
        committee.calibrate_uncertainty(X_val, y_val, method=calibration_method)
    y_pred_mean, y_pred_std = committee.predict(X_val, return_std=True)
    param_paths = [
        tmp_path / f"committee_model_{i}.json" for i in range(committee.n_models)
    ]
    serial_paths = [
        tmp_path / f"committee_model_{i}.pkl" for i in range(committee.n_models)
    ]
    committee.serialize(
        param_paths, serial_paths, calibration_path=calibration_model_path
    )
    committee = committee.deserialize(
        param_paths,
        serial_paths,
        mod_class=DummyRegressorModel,
        calibration_path=calibration_model_path,
    )
    y_pred_mean2, y_pred_std2 = committee.predict(X_val, return_std=True)
    assert_allclose(y_pred_mean, y_pred_mean2)
    assert_allclose(y_pred_std, y_pred_std2)
    assert committee.calibrated is (calibration_method is not None)


def test_plot_uncertainty_calibration(trained_committee):
    committee, X_val, y_val = trained_committee
    committee.calibrate_uncertainty(X_val, y_val, method="scaling-factor")
    plot = committee.plot_uncertainty_calibration(X_val, y_val)
    assert plot is not None
