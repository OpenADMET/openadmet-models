"""Tests for the inference orchestration pipeline using real, lightweight components."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openadmet.models.active_learning.committee import CommitteeRegressor
from openadmet.models.anvil.specification import DataSpec, Metadata
from openadmet.models.architecture.dummy import DummyRegressorModel
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.inference import inference as inference_module


@pytest.fixture
def input_df():
    """Provide a simple DataFrame with SMILES for testing inference inputs."""
    return pd.DataFrame({"MY_SMILES": ["CCO", "CCN"]})


@pytest.fixture(scope="module")
def real_featurizer():
    """Return a real FingerprintFeaturizer using ECFP4 fingerprints."""
    return FingerprintFeaturizer(fp_type="ecfp:4")


@pytest.fixture(scope="module")
def real_data_spec():
    """Return a real DataSpec with a single regression target."""
    return DataSpec(type="csv", target_cols=["task_0"], input_col="MY_SMILES")


@pytest.fixture(scope="module")
def real_metadata_single():
    """Return real Metadata with tag UNIT for single-model tests."""
    return Metadata(
        version="v1",
        driver="sklearn",
        name="unit-test",
        build_number=0,
        description="Unit test model",
        tag="UNIT",
        authors="Test Author",
        email="test@example.com",
        biotargets=["test"],
        tags=["test"],
    )


@pytest.fixture(scope="module")
def real_metadata_ensemble():
    """Return real Metadata with tag ENS for ensemble tests."""
    return Metadata(
        version="v1",
        driver="sklearn",
        name="ens-test",
        build_number=0,
        description="Ensemble test model",
        tag="ENS",
        authors="Test Author",
        email="test@example.com",
        biotargets=["test"],
        tags=["test"],
    )


@pytest.fixture(scope="module")
def trained_single_model():
    """Return a DummyRegressorModel trained to always predict 1.0 regardless of input features."""
    X_train = np.zeros((3, 2))
    y_train = np.array([[1.0], [1.0], [1.0]])
    model = DummyRegressorModel()
    model.train(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def trained_ensemble():
    """Return a CommitteeRegressor whose two members predict 1.0 and 3.0 respectively.

    The ensemble mean is 2.0 and the standard deviation is 1.0 for any input,
    making the UCB score with beta=2.0 equal to 4.0.
    """
    X_train = np.zeros((3, 2))

    model1 = DummyRegressorModel()
    model1.train(X_train, np.array([[1.0], [1.0], [1.0]]))

    model2 = DummyRegressorModel()
    model2.train(X_train, np.array([[3.0], [3.0], [3.0]]))

    return CommitteeRegressor.from_models([model1, model2])


def test_predict_with_real_single_model(
    mocker,
    input_df,
    real_featurizer,
    real_metadata_single,
    real_data_spec,
    trained_single_model,
):
    """Test the inference pipeline with a real DummyRegressorModel.

    SMILES strings flow through a real FingerprintFeaturizer and a real DummyRegressorModel
    to verify internal data plumbing. Because DummyRegressorModel always predicts the
    training mean, PRED values must equal 1.0 for both inputs. The STD column must be NaN
    because non-ensemble models produce no uncertainty estimate.
    """
    mock_loader = mocker.patch.object(
        inference_module,
        "load_anvil_model_and_metadata",
        return_value=(
            trained_single_model,
            real_featurizer,
            real_metadata_single,
            real_data_spec,
        ),
    )

    result = inference_module.predict(
        input_path=input_df,
        input_col="MY_SMILES",
        model_dir=["unused-model-dir"],
        accelerator="cpu",
        log=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert "OADMET_PRED_UNIT_task_0" in result.columns
    assert "OADMET_STD_UNIT_task_0" in result.columns
    assert result["OADMET_PRED_UNIT_task_0"].tolist() == pytest.approx([1.0, 1.0])
    assert result["OADMET_STD_UNIT_task_0"].isna().all()
    mock_loader.assert_called_once_with(Path("unused-model-dir"))


def test_predict_with_real_ensemble_and_acquisition(
    mocker,
    input_df,
    real_featurizer,
    real_metadata_ensemble,
    real_data_spec,
    trained_ensemble,
):
    """Test the inference pipeline with a real CommitteeRegressor and UCB acquisition.

    Two DummyRegressorModel members predict 1.0 and 3.0 respectively, yielding a committee
    mean of 2.0 and standard deviation of 1.0 for any input. With beta=2.0,
    UCB = mean + beta * std = 2.0 + 2.0 * 1.0 = 4.0.
    """
    mock_loader = mocker.patch.object(
        inference_module,
        "load_anvil_model_and_metadata",
        return_value=(
            trained_ensemble,
            real_featurizer,
            real_metadata_ensemble,
            real_data_spec,
        ),
    )

    result = inference_module.predict(
        input_path=input_df,
        input_col="MY_SMILES",
        model_dir=["unused-model-dir"],
        accelerator="cpu",
        log=False,
        aq_fxn_args={"ucb": {"beta": 2.0}},
    )

    assert result["OADMET_PRED_ENS_task_0"].tolist() == pytest.approx([2.0, 2.0])
    assert result["OADMET_STD_ENS_task_0"].tolist() == pytest.approx([1.0, 1.0])
    assert result["OADMET_UCB_ENS_task_0"].tolist() == pytest.approx([4.0, 4.0])
    mock_loader.assert_called_once_with(Path("unused-model-dir"))


def test_predict_raises_when_input_column_missing(input_df):
    """Ensure that the inference function validates the existence of the specified SMILES column."""
    with pytest.raises(ValueError, match="Column OTHER not found"):
        inference_module.predict(
            input_path=input_df,
            input_col="OTHER",
            model_dir=["unused-model-dir"],
            log=False,
        )


def test_load_anvil_model_and_metadata_missing_recipe_components(tmp_path):
    """Ensure correct error is raised when the model directory structure is invalid."""
    with pytest.raises(FileNotFoundError, match="does not contain recipe components"):
        inference_module.load_anvil_model_and_metadata(tmp_path)


def test_load_anvil_model_and_metadata_missing_procedure_yaml(tmp_path):
    """Ensure correct error is raised when critical YAML metadata files are missing."""
    model_dir = tmp_path / "model"
    recipe_components = model_dir / "recipe_components"
    recipe_components.mkdir(parents=True)
    (recipe_components / "metadata.yaml").write_text("metadata")
    (recipe_components / "data.yaml").write_text("data")

    with pytest.raises(FileNotFoundError, match="does not contain procedure.yaml"):
        inference_module.load_anvil_model_and_metadata(model_dir)
