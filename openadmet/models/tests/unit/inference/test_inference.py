"""Tests for the inference orchestration pipeline using real, lightweight components."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from openadmet.models.architecture.lgbm import LGBMRegressorModel
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.inference import inference as inference_module
from openadmet.models.transforms.pca import PCATransform
from openadmet.models.transforms.transform_base import transform_features
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
        _, feat, _, _, _ = inference_module.load_anvil_model_and_metadata(model_dir)

    assert feat.shuffle is False


# --- Fitted transform loading and application at inference ---


def _write_pca_recipe(model_dir):
    """
    Write recipe components for a PCA-reduced fingerprint LGBM model.

    The procedure declares a PCATransform, so loading requires the fitted
    transform artifact that a workflow run would have saved.
    """
    recipe_components = model_dir / "recipe_components"
    recipe_components.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": "v1",
        "driver": "sklearn",
        "name": "unit-test",
        "build_number": 0,
        "description": "Unit test model",
        "tag": "PCA",
        "authors": "Test Author",
        "email": "test@test.com",
        "biotargets": ["test"],
        "tags": ["test"],
    }
    with open(recipe_components / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)

    data_spec = {
        "type": "csv",
        "input_col": "MY_SMILES",
        "target_cols": ["task_0"],
    }
    with open(recipe_components / "data.yaml", "w") as f:
        yaml.safe_dump(data_spec, f)

    procedure = {
        "feat": {
            "type": "FingerprintFeaturizer",
            "params": {"fp_type": "ecfp", "n_jobs": 1},
        },
        "model": {
            "type": "LGBMRegressorModel",
            "params": {"n_estimators": 2, "num_leaves": 2, "random_seed": 42},
        },
        "split": {
            "type": "ShuffleSplitter",
            "params": {"train_size": 0.7, "test_size": 0.3, "random_seed": 42},
        },
        "train": {"type": "SKLearnBasicTrainer", "params": {}},
        "transform": {
            "type": "PCATransform",
            "params": {"n_components": 4, "random_seed": 42},
        },
    }
    with open(recipe_components / "procedure.yaml", "w") as f:
        yaml.safe_dump(procedure, f)


@pytest.fixture(scope="module")
def pca_recipe_dir(tmp_path_factory):
    """A model dir whose recipe declares a transform but has no fitted artifact."""
    model_dir = tmp_path_factory.mktemp("pca_recipe")
    _write_pca_recipe(model_dir)
    return model_dir


@pytest.fixture(scope="module")
def pca_model_dir(pca_recipe_dir, tmp_path_factory):
    """A model dir with an LGBM trained in PCA space plus the saved fitted transform."""
    import sklearn

    model_dir = tmp_path_factory.mktemp("pca_model")
    shutil.copytree(pca_recipe_dir, model_dir, dirs_exist_ok=True)

    smiles = ["CCO", "CCN", "CC(=O)OC", "c1ccccc1", "CCCCO"]
    feats, _ = FingerprintFeaturizer(fp_type="ecfp", n_jobs=1).featurize(smiles)
    pca = PCATransform(n_components=4, random_seed=42).fit(feats)

    with open(model_dir / "transform.pickle", "wb") as f:
        joblib.dump(
            {
                "schema": "v1",
                "transforms": [pca],
                "sklearn_version": sklearn.__version__,
            },
            f,
        )

    y = np.array([5.8, 5.6, 5.4, 5.2, 5.0])
    model = LGBMRegressorModel(n_estimators=2, num_leaves=2, random_seed=42)
    model.train(pca.transform(feats), y)
    model.serialize(
        model_dir / model._model_json_name, model_dir / (model._model_save_name)
    )
    return model_dir


def test_load_returns_fitted_transform(pca_model_dir):
    """Loading a model with a recipe transform must return the saved fitted sequence."""
    _, feat, transform, _, _ = inference_module.load_anvil_model_and_metadata(
        pca_model_dir
    )
    assert isinstance(transform, list)
    assert len(transform) == 1
    loaded_pca = transform[0]

    probe_feats, _ = FingerprintFeaturizer(fp_type="ecfp", n_jobs=1).featurize(["CCO"])
    out = loaded_pca.transform(np.atleast_2d(probe_feats))
    assert out.shape[1] == 4


def test_loader_missing_transform_artifact_raises(pca_recipe_dir):
    """A recipe transform without the fitted artifact must fail loudly, not silently skip."""
    with pytest.raises(ValueError, match="was trained with a transform"):
        inference_module.load_anvil_model_and_metadata(pca_recipe_dir)


def test_loader_rejects_unknown_transform_schema(pca_model_dir, tmp_path):
    """A transform artifact with an unrecognized schema must be rejected."""
    copied = tmp_path / "model"
    shutil.copytree(pca_model_dir, copied)
    with open(copied / "transform.pickle", "wb") as f:
        joblib.dump({"schema": "v99", "transforms": []}, f)

    with pytest.raises(ValueError, match="schema"):
        inference_module.load_anvil_model_and_metadata(copied)


def test_predict_single_row_applies_transform(pca_model_dir):
    """A single-row input (1D featurizer output) must survive the fitted transform path."""
    input_df = pd.DataFrame({"MY_SMILES": ["CCO"]})
    result = inference_module.predict(
        input_path=input_df,
        input_col="MY_SMILES",
        model_dir=[pca_model_dir],
        accelerator="cpu",
        log=False,
    )
    preds = result["OADMET_PRED_PCA_task_0"].to_numpy()
    assert preds.shape == (1,)


def test_predict_applies_transform_end_to_end(pca_model_dir):
    """predict() must apply the saved fitted transform, since the model was trained in PCA space."""
    input_df = pd.DataFrame({"MY_SMILES": ["CCO", "CCN"]})
    result = inference_module.predict(
        input_path=input_df,
        input_col="MY_SMILES",
        model_dir=[pca_model_dir],
        accelerator="cpu",
        log=False,
    )
    preds = result["OADMET_PRED_PCA_task_0"].to_numpy()

    # Independently: featurize, apply the same loaded transform, predict
    model, feat, transform, _, _ = inference_module.load_anvil_model_and_metadata(
        pca_model_dir
    )
    X, _ = feat.featurize(input_df["MY_SMILES"])
    X_transformed = transform_features(transform, X)
    expected = model.predict(X_transformed, accelerator="cpu")

    assert X.shape[1] != X_transformed.shape[1]
    np.testing.assert_allclose(preds, expected.ravel(), rtol=1e-12)
