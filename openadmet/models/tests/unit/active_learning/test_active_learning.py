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

    return model_list, X_feat, y


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

    return model_list, X_feat, y


@pytest.mark.parametrize(
    "model_list, calibration_method, query_strategy",
    product(
        ["lgbm_models", "chemprop_models"],
        ["isotonic-regression", "scaling-factor", None],
        _QUERY_STRATEGIES.keys(),
    ),
)
def test_committee(request, model_list, calibration_method, query_strategy):
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


@pytest.mark.parametrize("model_list", ["lgbm_models", "chemprop_models"])
def test_save_load(request, tmp_path, model_list):
    # Unpack models, features
    model_list, X_feat, y = request.getfixturevalue(model_list)

    # Create committee
    committee = CommitteeRegressor.from_models(models=model_list)

    # Predict before saving
    preds = committee.predict(X_feat, accelerator="cpu")

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
    preds2 = committee.predict(X_feat, accelerator="cpu")

    # Check that predictions are the same
    assert_allclose(preds, preds2)


@pytest.mark.parametrize("model_list", ["lgbm_models", "chemprop_models"])
def test_serialization(request, tmp_path, model_list):
    # Unpack models, features
    model_list, X_feat, y = request.getfixturevalue(model_list)

    # Create committee
    committee = CommitteeRegressor.from_models(models=model_list)

    # Predict before saving
    preds = committee.predict(X_feat, accelerator="cpu")

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
    preds2 = committee.predict(X_feat, accelerator="cpu")

    # Check that predictions are the same
    assert_allclose(preds, preds2)
