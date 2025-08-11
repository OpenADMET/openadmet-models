from itertools import product
from pathlib import Path

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from openadmet.models.active_learning.acquisition import _QUERY_STRATEGIES
from openadmet.models.active_learning.committee import (
    CommitteeRegressor,
)
from openadmet.models.inference.inference import load_anvil_model_and_metadata
from openadmet.models.tests.unit.datafiles import (
    ACEH_chembl_pchembl,  # chemprop
    CYP3A4_chembl_pchembl,  # lgbm
    anvil_chemprop_trained_model_dir,
    anvil_lgbm_trained_model_dir,
)


@pytest.fixture
def chemprop_models():
    # Load the model and metadata
    models = []
    for i in range(5):
        model, feat, _, _ = load_anvil_model_and_metadata(
            Path(anvil_chemprop_trained_model_dir)
        )
        models.append(model)

    # Load data
    data = pd.read_csv(ACEH_chembl_pchembl).iloc[:100, :]
    X = data["OPENADMET_SMILES"].values
    y = data["pchembl_value_mean"].values

    # Featurize
    X_feat = feat.featurize(X)[0]

    return models, X_feat, y


@pytest.fixture
def lgbm_models():
    models = []
    for i in range(5):
        model, feat, _, _ = load_anvil_model_and_metadata(
            Path(anvil_lgbm_trained_model_dir)
        )
        models.append(model)

    # Load data
    data = pd.read_csv(CYP3A4_chembl_pchembl).iloc[:100, :]
    X = data["CANONICAL_SMILES"].values
    y = data["pChEMBL mean"].values

    # Featurize
    X_feat = feat.featurize(X)[0]

    return models, X_feat, y


@pytest.mark.parametrize(
    "query_strategy, model_list, calibration_method",
    product(
        _QUERY_STRATEGIES.keys(),
        ["lgbm_models", "chemprop_models"],
        ["isotonic-regression", "scaling-factor", None],
    ),
)
def test_committee(query_strategy, model_list, calibration_method, request):
    # Unpack models, features
    model_list, X, y = request.getfixturevalue(model_list)

    # Create committee
    committee = CommitteeRegressor.from_models(models=model_list)

    # Calibrate uncertainty
    if calibration_method is not None:
        committee.calibrate_uncertainty(X, y, method=calibration_method)

    # Query
    y_query = committee.query(X, query_strategy=query_strategy, accelerator="cpu")

    # Predict
    y_pred, y_pred_std = committee.predict(X, return_std=True, accelerator="cpu")


@pytest.mark.parametrize("model_list", ["lgbm_models", "chemprop_models"])
def test_save_load(tmp_path, model_list, request):
    # Unpack models, features
    model_list, X, y = request.getfixturevalue(model_list)

    # Create committee
    committee = CommitteeRegressor.from_models(models=model_list)

    # Predict before saving
    preds = committee.predict(X, accelerator="cpu")

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
    preds2 = committee.predict(X, accelerator="cpu")

    # Check that predictions are the same
    assert_allclose(preds, preds2)


@pytest.mark.parametrize("model_list", ["lgbm_models", "chemprop_models"])
def test_serialization(tmp_path, model_list, request):
    # Unpack models, features
    model_list, X, y = request.getfixturevalue(model_list)

    # Create committee
    committee = CommitteeRegressor.from_models(models=model_list)

    # Predict before saving
    preds = committee.predict(X, accelerator="cpu")

    # Save and load
    param_paths = [
        tmp_path / "committee_model_{i}.json" for i in range(len(model_list))
    ]
    serial_paths = [
        tmp_path / "committee_model_{i}.pkl" for i in range(len(model_list))
    ]
    committee.serialize(param_paths, serial_paths)
    committee.deserialize(param_paths, serial_paths, mod_class=model_list[0].__class__)

    # Predict after loading
    preds2 = committee.predict(X, accelerator="cpu")

    # Check that predictions are the same
    assert_allclose(preds, preds2)
