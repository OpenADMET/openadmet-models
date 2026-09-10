"""Session-scoped fixtures providing lightweight on-disk model directories for unit tests."""

import joblib
import yaml
import numpy as np
import pytest

from openadmet.models.architecture.dummy import DummyRegressorModel


def _write_recipe_components(
    recipe_dir,
    tag,
    ensemble=False,
    name="unit-test",
    feat=None,
    model=None,
    transform=None,
):
    """
    Write the three required YAML files into a recipe_components directory.

    Defaults to a NullFeaturizer, which keeps every input row. Pass `feat` to
    use a featurizer that drops rows, which is what exercises index propagation.
    Pass `model` for an architecture that is sensitive to feature width, and
    `transform` to declare a transform the loader then expects on disk.
    """
    recipe_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": "v1",
        "driver": "sklearn",
        "name": name,
        "build_number": 0,
        "description": f"Unit test model ({tag})",
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
        "feat": feat or {"type": "NullFeaturizer", "params": {}},
        "model": model or {"type": "DummyRegressorModel", "params": {}},
        "split": {
            "type": "ShuffleSplitter",
            "params": {"train_size": 0.8, "test_size": 0.2, "random_seed": 42},
        },
        "train": {"type": "SKLearnBasicTrainer", "params": {}},
    }
    if transform is not None:
        procedure["transform"] = transform

    if ensemble:
        procedure["ensemble"] = {
            "type": "CommitteeRegressor",
            "n_models": 2,
            "params": {},
        }

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


@pytest.fixture(scope="session")
def fingerprint_model_dir(tmp_path_factory):
    """
    Session-scoped on-disk model directory whose featurizer drops bad SMILES.

    The NullFeaturizer used by the other fixtures keeps every row, so it cannot
    exercise index propagation. This model always predicts 5.0 regardless of
    input features (tag=FP, target=task_0).
    """
    model_dir = tmp_path_factory.mktemp("fingerprint_model")
    _write_recipe_components(
        model_dir / "recipe_components",
        tag="FP",
        name="unit-test-fp",
        feat={
            "type": "FingerprintFeaturizer",
            "params": {"fp_type": "ecfp", "n_jobs": 1},
        },
    )

    model = _make_trained_dummy(5.0)
    model.serialize(model_dir / "model.json", model_dir / "model.pkl")

    return model_dir


@pytest.fixture(scope="session")
def transform_model_dir(tmp_path_factory):
    """
    Session-scoped on-disk model directory whose recipe carries a fitted PCA transform.

    The LGBM inside is trained in PCA space, so it only accepts the reduced
    width. Anything that feeds it raw fingerprints instead of running the saved
    transform first fails on the feature count (tag=TFM, target=task_0).
    """
    import sklearn

    from openadmet.models.architecture.lgbm import LGBMRegressorModel
    from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
    from openadmet.models.transforms.pca import PCATransform

    model_dir = tmp_path_factory.mktemp("transform_model")
    _write_recipe_components(
        model_dir / "recipe_components",
        tag="TFM",
        name="unit-test-tfm",
        feat={
            "type": "FingerprintFeaturizer",
            "params": {"fp_type": "ecfp", "n_jobs": 1},
        },
        model={
            "type": "LGBMRegressorModel",
            "params": {"n_estimators": 2, "num_leaves": 2, "random_seed": 42},
        },
        transform={
            "type": "PCATransform",
            "params": {"n_components": 4, "random_seed": 42},
        },
    )

    # Fit the transform the workflow would have saved next to the model
    train_smiles = ["CCO", "CCN", "CC(=O)OC", "c1ccccc1", "CCCCO"]
    feats, _ = FingerprintFeaturizer(fp_type="ecfp", n_jobs=1).featurize(train_smiles)
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

    model = LGBMRegressorModel(n_estimators=2, num_leaves=2, random_seed=42)
    model.train(pca.transform(feats), np.array([5.8, 5.6, 5.4, 5.2, 5.0]))
    model.serialize(model_dir / "model.json", model_dir / "model.pkl")

    return model_dir


@pytest.fixture(scope="session")
def chemeleon_foundation_checkpoint(tmp_path_factory):
    """
    Session-scoped foundation checkpoint with the CheMeleon layout and random weights.

    Lets tests take the real from_foundation path without downloading the published
    CheMeleon checkpoint. Weights are random, so only shapes and determinism are
    meaningful, never the embedding values themselves.
    """
    # Imported here so the whole unit suite does not pay for torch and chemprop
    import torch
    from chemprop import nn

    from openadmet.models.architecture.chemprop import _CHEMELEON_MP_HPARAMS

    message_passing = nn.BondMessagePassing(**_CHEMELEON_MP_HPARAMS)
    checkpoint_path = (
        tmp_path_factory.mktemp("chemeleon_foundation") / "chemeleon_mp.pt"
    )
    torch.save(
        {
            "hyper_parameters": dict(_CHEMELEON_MP_HPARAMS),
            "state_dict": message_passing.state_dict(),
        },
        checkpoint_path,
    )

    return checkpoint_path
