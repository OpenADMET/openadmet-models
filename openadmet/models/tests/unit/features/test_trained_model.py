"""Unit tests for TrainedModelFeaturizer."""

import numpy as np
import pytest
from pydantic import ValidationError

from openadmet.models.features.combine import FeatureConcatenator
from openadmet.models.features.feature_base import get_featurizer_class
from openadmet.models.features.trained_model import TrainedModelFeaturizer
from openadmet.models.inference.inference import load_anvil_model_and_metadata
from openadmet.models.transforms.transform_base import transform_features


@pytest.fixture(scope="module")
def smiles():
    """Provide valid SMILES strings for featurization."""
    return ["CCO", "CCN", "c1ccccc1"]


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


def test_include_std_appends_the_stdev_after_the_prediction(
    null_ensemble_model_dir, smiles
):
    """include_std must widen the block, with the stdev columns after the predictions."""
    feat = TrainedModelFeaturizer(model_dir=null_ensemble_model_dir, include_std=True)
    features, _ = feat.featurize(smiles)

    # Members 1.0 and 3.0 give a mean of 2.0 and a standard deviation of 1.0
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


def test_featurize_applies_the_pretrained_transform(transform_model_dir):
    """A pretrained model fitted in transform space must be predicted on in that space."""
    smiles = ["CCO", "CCN"]
    feat = TrainedModelFeaturizer(model_dir=transform_model_dir)
    features, _ = feat.featurize(smiles)

    # Reach the same predictions by transforming before predicting, as inference does
    model, pretrained_feat, transform, _, _ = load_anvil_model_and_metadata(
        transform_model_dir
    )
    X, _ = pretrained_feat.featurize(smiles)
    expected = model.predict(transform_features(transform, X), accelerator="cpu")

    np.testing.assert_allclose(features.ravel(), expected.ravel(), rtol=1e-12)


def test_std_from_a_non_ensemble_model_raises_at_construction(null_single_model_dir):
    """Requesting a stdev from a model that has none must fail before any featurization."""
    with pytest.raises(ValidationError, match="is not an ensemble"):
        TrainedModelFeaturizer(model_dir=null_single_model_dir, include_std=True)


def test_rejects_a_directory_that_is_not_a_model(tmp_path):
    """A missing directory, or one without recipe components, must fail at construction."""
    with pytest.raises(ValidationError, match="does not exist"):
        TrainedModelFeaturizer(model_dir=tmp_path / "absent")

    with pytest.raises(ValidationError, match="no recipe_components"):
        TrainedModelFeaturizer(model_dir=tmp_path)


def test_rejects_a_recipe_without_a_procedure(tmp_path):
    """A recipe missing procedure.yaml must fail at construction either way."""
    (tmp_path / "recipe_components").mkdir()

    # include_std defaults to False, so this fails without any stdev involvement
    with pytest.raises(ValidationError, match="no recipe_components/procedure.yaml"):
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
        featurizers=[
            {"type": "NullFeaturizer"},
            {
                "type": "TrainedModelFeaturizer",
                "params": {"model_dir": str(null_single_model_dir)},
            },
        ]
    )
    features, indices = concat.featurize(smiles)

    # Blocks land in the configured order, so the null block is first
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
