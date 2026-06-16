"""Tests for the inference orchestration pipeline using real, lightweight components."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from openadmet.models.inference import inference as inference_module
from openadmet.models.tests.unit.datafiles import anvil_chemprop_trained_model_dir


@pytest.fixture
def input_df():
    """Provide a simple DataFrame with SMILES for testing inference inputs."""
    return pd.DataFrame({"MY_SMILES": ["CCO", "CCN"]})


def test_predict_with_real_single_model(input_df, null_single_model_dir):
    """Test the inference pipeline end-to-end with a real on-disk DummyRegressorModel.

    SMILES strings are featurized by NullFeaturizer and passed to a DummyRegressorModel
    loaded from disk via load_anvil_model_and_metadata. Because DummyRegressorModel always
    predicts the training mean (1.0), PRED values must equal 1.0 for both inputs.
    The STD column must be NaN because non-ensemble models produce no uncertainty estimate.
    """
    result = inference_module.predict(
        input_path=input_df,
        input_col="MY_SMILES",
        model_dir=[null_single_model_dir],
        accelerator="cpu",
        log=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert "OADMET_PRED_UNIT_task_0" in result.columns
    assert "OADMET_STD_UNIT_task_0" in result.columns
    assert np.allclose(result["OADMET_PRED_UNIT_task_0"].to_numpy(), [1.0, 1.0])
    assert result["OADMET_STD_UNIT_task_0"].isna().all()


def test_predict_with_real_ensemble_and_acquisition(input_df, null_ensemble_model_dir):
    """Test the inference pipeline end-to-end with a real on-disk CommitteeRegressor.

    Two DummyRegressorModel members (predicting 1.0 and 3.0) are loaded from disk and
    assembled into a CommitteeRegressor. The ensemble mean is 2.0 and the standard
    deviation is 1.0 for any input. With beta=2.0, UCB = mean + beta * std = 4.0.
    """
    result = inference_module.predict(
        input_path=input_df,
        input_col="MY_SMILES",
        model_dir=[null_ensemble_model_dir],
        accelerator="cpu",
        log=False,
        aq_fxn_args={"ucb": {"beta": 2.0}},
    )

    assert np.allclose(result["OADMET_PRED_ENS_task_0"].to_numpy(), [2.0, 2.0])
    assert np.allclose(result["OADMET_STD_ENS_task_0"].to_numpy(), [1.0, 1.0])
    assert np.allclose(result["OADMET_UCB_ENS_task_0"].to_numpy(), [4.0, 4.0])


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


def test_load_anvil_forces_shuffle_false(tmp_path):
    """A featurizer saved with shuffle=True during training must have shuffle forced to False at load time."""
    model_dir = tmp_path / "model"
    shutil.copytree(anvil_chemprop_trained_model_dir, model_dir)

    procedure = {
        "feat": {"type": "ChemPropFeaturizer", "params": {"shuffle": True}},
        "model": {"type": "ChemPropModel", "params": {}},
        "split": {
            "type": "ShuffleSplitter",
            "params": {"random_seed": 42, "test_size": 0.3, "train_size": 0.7},
        },
        "train": {
            "type": "LightningTrainer",
            "params": {"gpus": 1, "max_epochs": 10, "use_wandb": False},
        },
    }
    (model_dir / "recipe_components" / "procedure.yaml").write_text(
        yaml.dump(procedure)
    )

    with patch(
        "openadmet.models.architecture.model_base.LightningModelBase.deserialize",
        return_value=MagicMock(),
    ):
        _, feat, _, _ = inference_module.load_anvil_model_and_metadata(model_dir)

    assert feat.shuffle is False
