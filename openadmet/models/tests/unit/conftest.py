"""Session-scoped fixtures providing lightweight on-disk model directories for unit tests."""

import yaml
import numpy as np
import pytest

from openadmet.models.architecture.dummy import DummyRegressorModel


def _write_recipe_components(recipe_dir, tag, ensemble=False):
    """Write the three required YAML files into a recipe_components directory."""
    recipe_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": "v1",
        "driver": "sklearn",
        "name": "unit-test",
        "build_number": 0,
        "description": "Unit test model",
        "tag": tag,
        "authors": "Test Author",
        "email": "test@test.com",
        "biotargets": ["test"],
        "tags": ["test"],
    }
    with open(recipe_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)

    data_spec = {
        "type": "csv",
        "input_col": "MY_SMILES",
        "target_cols": ["task_0"],
    }
    with open(recipe_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_spec, f)

    procedure = {
        "feat": {"type": "NullFeaturizer", "params": {}},
        "model": {"type": "DummyRegressorModel", "params": {}},
        "split": {
            "type": "ShuffleSplitter",
            "params": {"train_size": 0.8, "test_size": 0.2, "random_state": 42},
        },
        "train": {"type": "SKLearnBasicTrainer", "params": {}},
    }
    if ensemble:
        procedure["ensemble"] = {"type": "CommitteeRegressor", "n_models": 2, "params": {}}

    with open(recipe_dir / "procedure.yaml", "w") as f:
        yaml.safe_dump(procedure, f)


def _make_trained_dummy(constant_value):
    """Return a DummyRegressorModel trained to always predict `constant_value`."""
    X_train = np.zeros((3, 1))
    y_train = np.full(3, constant_value)
    model = DummyRegressorModel()
    model.train(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def null_single_model_dir(tmp_path_factory):
    """
    Session-scoped on-disk model directory for a single DummyRegressorModel.

    The directory contains a complete recipe_components layout plus serialized
    model files. The model always predicts 1.0 regardless of input features
    (tag=UNIT, target=task_0).
    """
    model_dir = tmp_path_factory.mktemp("null_single_model")
    _write_recipe_components(model_dir / "recipe_components", tag="UNIT")

    model = _make_trained_dummy(1.0)
    model.serialize(model_dir / "model.json", model_dir / "model.pkl")

    return model_dir


@pytest.fixture(scope="session")
def null_ensemble_model_dir(tmp_path_factory):
    """
    Session-scoped on-disk model directory for a two-member CommitteeRegressor.

    Member 0 predicts 1.0 and member 1 predicts 3.0, so the ensemble mean is 2.0
    and the standard deviation is 1.0 for any input (tag=ENS, target=task_0).
    """
    model_dir = tmp_path_factory.mktemp("null_ensemble_model")
    _write_recipe_components(model_dir / "recipe_components", tag="ENS", ensemble=True)

    for i, constant_value in enumerate([1.0, 3.0]):
        member_dir = model_dir / f"model_{i}"
        member_dir.mkdir()
        model = _make_trained_dummy(constant_value)
        model.serialize(member_dir / "model.json", member_dir / "model.pkl")

    return model_dir
