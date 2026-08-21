"""Unit tests for TrainedModelFeaturizer."""

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from openadmet.models.architecture.dummy import DummyRegressorModel
from openadmet.models.features.combine import FeatureConcatenator
from openadmet.models.features.feature_base import get_featurizer_class
from openadmet.models.features.trained_model import TrainedModelFeaturizer


@pytest.fixture(scope="module")
def smiles():
    """Provide valid SMILES strings for featurization."""
    return ["CCO", "CCN", "c1ccccc1"]


@pytest.fixture(scope="module")
def fingerprint_model_dir(tmp_path_factory):
    """
    Provide a model directory whose featurizer drops unparseable SMILES.

    The NullFeaturizer used by the shared fixtures keeps every row, so it cannot
    exercise index propagation. This model predicts 5.0 for any input
    (tag=FP, target=task_0).
    """
    model_dir = tmp_path_factory.mktemp("fingerprint_model")
    recipe_dir = model_dir / "recipe_components"
    recipe_dir.mkdir(parents=True)

    with open(recipe_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(
            {
                "version": "v1",
                "driver": "sklearn",
                "name": "unit-test-fp",
                "build_number": 0,
                "description": "Unit test fingerprint model",
                "tag": "FP",
                "authors": "Test Author",
                "email": "test@test.com",
                "biotargets": ["test"],
                "tags": ["test"],
            },
            f,
        )
    with open(recipe_dir / "data.yaml", "w") as f:
        yaml.safe_dump(
            {"type": "csv", "input_col": "MY_SMILES", "target_cols": ["task_0"]}, f
        )
    with open(recipe_dir / "procedure.yaml", "w") as f:
        yaml.safe_dump(
            {
                "feat": {
                    "type": "FingerprintFeaturizer",
                    "params": {"fp_type": "ecfp", "n_jobs": 1},
                },
                "model": {"type": "DummyRegressorModel", "params": {}},
                "split": {
                    "type": "ShuffleSplitter",
                    "params": {"train_size": 0.8, "test_size": 0.2, "random_seed": 42},
                },
                "train": {"type": "SKLearnBasicTrainer", "params": {}},
            },
            f,
        )

    model = DummyRegressorModel()
    model.train(np.zeros((3, 1)), np.full(3, 5.0))
    model.serialize(model_dir / "model.json", model_dir / "model.pkl")

    return model_dir


def test_featurize_emits_the_models_predictions(null_single_model_dir, smiles):
    """A single-task model must emit one column carrying its prediction for every row."""
    feat = TrainedModelFeaturizer(model_dir=null_single_model_dir)
    features, indices = feat.featurize(smiles)

    # The fixture model predicts 1.0 regardless of input
    np.testing.assert_array_equal(features, np.ones((len(smiles), 1)))
    np.testing.assert_array_equal(indices, np.arange(len(smiles)))


def test_featurize_emits_the_ensemble_mean(null_ensemble_model_dir, smiles):
    """An ensemble must contribute its mean, not any individual member's prediction."""
    feat = TrainedModelFeaturizer(model_dir=null_ensemble_model_dir)
    features, _ = feat.featurize(smiles)

    # Members predict 1.0 and 3.0, so the mean is 2.0
    np.testing.assert_array_equal(features, np.full((len(smiles), 1), 2.0))


def test_featurize_emits_outputs_in_configured_order(null_ensemble_model_dir, smiles):
    """Requesting mean and std must widen the block, in the order the outputs are listed."""
    feat = TrainedModelFeaturizer(
        model_dir=null_ensemble_model_dir, outputs=["mean", "std"]
    )
    features, _ = feat.featurize(smiles)

    # Members 1.0 and 3.0 give a mean of 2.0 and a spread of 1.0
    assert features.shape == (len(smiles), 2)
    np.testing.assert_array_equal(features[:, 0], np.full(len(smiles), 2.0))
    np.testing.assert_array_equal(features[:, 1], np.full(len(smiles), 1.0))


def test_featurize_reports_the_rows_its_featurizer_kept(fingerprint_model_dir):
    """Molecules the pretrained featurizer drops must be absent from both features and indices."""
    with_invalid = ["CCO", "not_a_molecule", "CCN"]
    feat = TrainedModelFeaturizer(model_dir=fingerprint_model_dir)
    features, indices = feat.featurize(with_invalid)

    # The unparseable entry at position 1 survives in neither output
    np.testing.assert_array_equal(indices, np.array([0, 2]))
    np.testing.assert_array_equal(features, np.full((2, 1), 5.0))


def test_std_from_a_non_ensemble_model_raises_at_construction(null_single_model_dir):
    """Requesting a spread from a model that has none must fail before any featurization."""
    with pytest.raises(ValidationError, match="is not an ensemble"):
        TrainedModelFeaturizer(model_dir=null_single_model_dir, outputs=["mean", "std"])


@pytest.mark.parametrize(
    "outputs, match",
    [
        pytest.param([], "at least 1 item", id="empty"),
        pytest.param(["mean", "mean"], "Duplicate outputs", id="repeated"),
        pytest.param(["variance"], "Input should be", id="unknown"),
    ],
)
def test_rejects_invalid_outputs(null_single_model_dir, outputs, match):
    """Outputs must be a non-empty list of distinct known quantities."""
    with pytest.raises(ValidationError, match=match):
        TrainedModelFeaturizer(model_dir=null_single_model_dir, outputs=outputs)


def test_rejects_a_directory_that_is_not_a_model(tmp_path):
    """A missing directory, or one without recipe components, must fail at construction."""
    with pytest.raises(ValidationError, match="does not exist"):
        TrainedModelFeaturizer(model_dir=tmp_path / "absent")

    with pytest.raises(ValidationError, match="no recipe_components"):
        TrainedModelFeaturizer(model_dir=tmp_path)


def test_registered_under_its_type(null_single_model_dir):
    """The featurizer must be reachable through the registry, as a recipe would reach it."""
    feat_class = get_featurizer_class("TrainedModelFeaturizer")
    assert feat_class is TrainedModelFeaturizer

    feat = feat_class(model_dir=null_single_model_dir)
    assert feat.type == "TrainedModelFeaturizer"


def test_composes_inside_a_concatenator(null_single_model_dir, smiles):
    """Predictions must concatenate alongside ordinary features as one more block of columns."""
    concat = FeatureConcatenator(
        featurizers={
            "NullFeaturizer": {},
            "TrainedModelFeaturizer": {"model_dir": str(null_single_model_dir)},
        }
    )
    features, indices = concat.featurize(smiles)

    # Class-name order puts the null block first, so its zero column precedes
    # the trained model's 1.0 column
    assert features.shape == (len(smiles), 2)
    np.testing.assert_array_equal(indices, np.arange(len(smiles)))
    np.testing.assert_array_equal(features[:, 0], np.zeros(len(smiles)))
    np.testing.assert_array_equal(features[:, 1], np.ones(len(smiles)))


def test_model_is_loaded_once_across_calls(null_single_model_dir, smiles):
    """The trained model must be deserialized once, not per partition featurized."""
    feat = TrainedModelFeaturizer(model_dir=null_single_model_dir)
    feat.featurize(smiles)
    loaded_model = feat._loaded[0]

    feat.featurize(smiles)
    assert feat._loaded[0] is loaded_model
