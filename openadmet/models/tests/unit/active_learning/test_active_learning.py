from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from openadmet.models.active_learning.acquisition import _ACQUISITION_FUNCTIONS
from openadmet.models.active_learning.committee import (
    CommitteeRegressor,
)
from openadmet.models.architecture.lgbm import LGBMRegressorModel
from openadmet.models.inference.inference import load_anvil_model_and_metadata
from openadmet.models.split.sklearn import ShuffleSplitter
from openadmet.models.tests.unit.datafiles import (
    ACEH_chembl_pchembl,  # chemprop
    CYP3A4_chembl_pchembl,  # lgbm
    anvil_chemprop_trained_model_dir,
    anvil_lgbm_trained_model_dir,
)


@pytest.fixture
def chemprop_models():
    # Load the model and metadata
    model_list = []
    for i in range(5):
        model, feat, _, _ = load_anvil_model_and_metadata(
            Path(anvil_chemprop_trained_model_dir)
        )
        model_list.append(model)

    # Load data
    data = pd.read_csv(ACEH_chembl_pchembl).iloc[:100, :]
    X = data["OPENADMET_SMILES"].values
    y = data["pchembl_value_mean"].values

    # Featurize
    X_feat = feat.featurize(X)[0]

    return model_list, X_feat, y.reshape(-1, 1)


@pytest.fixture
def lgbm_models():
    model_list = []
    for i in range(5):
        model, feat, _, _ = load_anvil_model_and_metadata(
            Path(anvil_lgbm_trained_model_dir)
        )
        model_list.append(model)

    # Load data
    data = pd.read_csv(CYP3A4_chembl_pchembl).iloc[:100, :]
    X = data["CANONICAL_SMILES"].values
    y = data["pChEMBL mean"].values

    # Featurize
    X_feat = feat.featurize(X)[0]

    return model_list, X_feat, y.reshape(-1, 1)


@pytest.fixture
def toy_data():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of samples
    n_samples = 2000

    # Features
    X = np.column_stack(
        [
            np.linspace(0, 10, n_samples),
            np.random.uniform(0, 5, n_samples),
            np.random.normal(5, 2, n_samples),
        ]
    )

    # Targets
    y = np.column_stack(
        [
            3 * np.sin(X[:, 0])
            + 0.5 * X[:, 1] ** 2
            - 0.8 * X[:, 2]
            + np.random.normal(0, 0.1, n_samples),
            2 * np.cos(X[:, 0])
            + 0.3 * X[:, 1] ** 2
            + 0.5 * X[:, 2]
            + np.random.normal(0, 0.1, n_samples),
        ]
    )

    # Split the data
    splitter = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        X, y[:, 0].reshape(-1, 1)
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


@pytest.mark.parametrize(
    "model_list, calibration_method, query_strategy",
    product(
        ["lgbm_models", "chemprop_models"],
        ["isotonic-regression", "scaling-factor", None],
        _ACQUISITION_FUNCTIONS.keys(),
    ),
)
def test_committee(request, model_list, calibration_method, query_strategy):
    # Skip calibration on real data until we have true ensemble
    if calibration_method is not None:
        # Skip the test
        pytest.skip("Skipping calibration test")

    # Unpack models, features
    _model_list, X_feat, y = request.getfixturevalue(model_list)

    # Create committee
    committee = CommitteeRegressor.from_models(models=_model_list)

    # Calibrate uncertainty
    if calibration_method is not None:
        committee.calibrate_uncertainty(
            X_feat, y, method=calibration_method, accelerator="cpu"
        )

    # Query
    y_query = committee.query(X_feat, query_strategy=query_strategy, accelerator="cpu")

    # Predict
    y_pred, y_pred_std = committee.predict(X_feat, return_std=True, accelerator="cpu")


@pytest.mark.parametrize(
    "model_list, calibration_method",
    product(
        ["lgbm_models", "chemprop_models"],
        ["isotonic-regression", "scaling-factor", None],
    ),
)
def test_save_load(request, tmp_path, model_list, calibration_method):
    # Skip calibration on real data until we have true ensemble
    if calibration_method is not None:
        # Skip the test
        pytest.skip("Skipping calibration test")

    # Unpack models, features
    model_list, X_feat, y = request.getfixturevalue(model_list)

    # Create committee
    committee = CommitteeRegressor.from_models(models=model_list)

    # Calibrate uncertainty
    if calibration_method is not None:
        committee.calibrate_uncertainty(
            X_feat, y, method=calibration_method, accelerator="cpu"
        )

    # Predict before saving
    y_pred_mean, y_pred_std = committee.predict(
        X_feat, return_std=True, accelerator="cpu"
    )

    # Save
    save_paths = [tmp_path / "committee_model_{i}.pkl" for i in range(len(model_list))]
    committee.save(save_paths)

    # Instantiate empty models to "fill"
    models_new = [model.make_new() for model in model_list]
    [model.build() for model in models_new]

    # Load
    committee.load(
        save_paths,
        models=models_new,
    )

    # Predict after loading
    y_pred_mean2, y_pred_std2 = committee.predict(
        X_feat, return_std=True, accelerator="cpu"
    )

    # Check that predictions are the same
    assert_allclose(y_pred_mean, y_pred_mean2)
    assert_allclose(y_pred_std, y_pred_std2)


@pytest.mark.parametrize(
    "model_list, calibration_method",
    product(
        ["lgbm_models", "chemprop_models"],
        ["isotonic-regression", "scaling-factor", None],
    ),
)
def test_serialization(request, tmp_path, model_list, calibration_method):
    # Skip calibration on real data until we have true ensemble
    if calibration_method is not None:
        # Skip the test
        pytest.skip("Skipping calibration test")

    # Unpack models, features
    model_list, X_feat, y = request.getfixturevalue(model_list)

    # Create committee
    committee = CommitteeRegressor.from_models(models=model_list)

    # Calibrate uncertainty
    if calibration_method is not None:
        committee.calibrate_uncertainty(
            X_feat, y, method=calibration_method, accelerator="cpu"
        )

    # Predict before saving
    y_pred_mean, y_pred_std = committee.predict(
        X_feat, return_std=True, accelerator="cpu"
    )

    # Serialize/deserialize
    param_paths = [
        tmp_path / "committee_model_{i}.json" for i in range(len(model_list))
    ]
    serial_paths = [
        tmp_path / "committee_model_{i}.pkl" for i in range(len(model_list))
    ]
    committee.serialize(param_paths, serial_paths)
    committee.deserialize(param_paths, serial_paths, mod_class=model_list[0].__class__)

    # Predict after loading
    y_pred_mean2, y_pred_std2 = committee.predict(
        X_feat, return_std=True, accelerator="cpu"
    )

    # Check that predictions are the same
    assert_allclose(y_pred_mean, y_pred_mean2)
    assert_allclose(y_pred_std, y_pred_std2)


# This test is somewhat redundant and a catch-all, but useful until we have ability to test real-world ensembles
# more explicitly
@pytest.mark.parametrize(
    "calibration_method", ["isotonic-regression", "scaling-factor", None]
)
def test_calibration(tmp_path, toy_data, calibration_method):
    # Unpack data
    X_train, X_val, X_test, y_train, y_val, y_test = toy_data

    # Parameters
    mod_params = {"alpha": 0.005, "learning_rate": 0.05, "force_col_wise": True}

    # Train committee
    committee = CommitteeRegressor.train(
        X_train,
        y_train,
        mod_class=LGBMRegressorModel,
        mod_params=mod_params,
        n_models=5,
    )

    # Calibrate uncertainty
    if calibration_method is not None:
        committee.calibrate_uncertainty(X_val, y_val, method=calibration_method)

    # Evaluate on test set
    y_pred_mean, y_pred_std = committee.predict(X_test, return_std=True)

    # Generate plot
    committee.plot_uncertainty_calibration(X_test, y_test)

    # Serialize/deserialize
    param_paths = [
        tmp_path / "committee_model_{i}.json" for i in range(len(committee.models))
    ]
    serial_paths = [
        tmp_path / "committee_model_{i}.pkl" for i in range(len(committee.models))
    ]
    committee.serialize(param_paths, serial_paths)
    committee.deserialize(param_paths, serial_paths, mod_class=LGBMRegressorModel)

    # Evaluate on test set again
    y_pred_mean2, y_pred_std2 = committee.predict(X_test, return_std=True)

    # Check results match original
    assert_allclose(y_pred_mean, y_pred_mean2)
    assert_allclose(y_pred_std, y_pred_std2)

    # Check that we successfully loaded calibration models
    if calibration_method is not None:
        assert committee.calibrated

    # Save/load
    save_paths = [
        tmp_path / "committee_model_{i}.pkl" for i in range(len(committee.models))
    ]
    committee.save(save_paths)

    # Instantiate empty models to "fill"
    models_new = [
        LGBMRegressorModel.from_params(mod_params=mod_params)
        for _ in range(len(committee.models))
    ]

    # Load
    committee.load(
        save_paths,
        models=models_new,
    )

    # Evaluate on test set again
    y_pred_mean2, y_pred_std2 = committee.predict(X_test, return_std=True)

    # Check results match original
    assert_allclose(y_pred_mean, y_pred_mean2)
    assert_allclose(y_pred_std, y_pred_std2)

    # Check that we successfully loaded calibration models
    if calibration_method is not None:
        assert committee.calibrated
