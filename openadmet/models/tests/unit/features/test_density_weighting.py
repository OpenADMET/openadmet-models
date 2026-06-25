import numpy as np
import pytest

from openadmet.models.features.chemprop import (
    ChemPropFeaturizer,
    _inverse_density_weights,
)


@pytest.fixture
def skewed_targets():
    """Right-skewed single-task targets: a dense low cluster and a rare high tail."""
    rng = np.random.default_rng(0)
    dense = rng.normal(5.0, 0.3, 400)
    tail = rng.normal(8.0, 0.3, 20)
    return np.concatenate([dense, tail]).reshape(-1, 1)


def test_weights_normalized_to_unit_mean(skewed_targets):
    """Per-sample weights are normalized so their mean is 1, preserving loss scale."""
    weights = _inverse_density_weights(skewed_targets)

    assert weights.mean() == pytest.approx(1.0)


def test_tail_samples_weighted_above_dense_samples(skewed_targets):
    """Rare tail targets receive larger weights than the dense majority."""
    weights = _inverse_density_weights(skewed_targets)
    is_tail = skewed_targets[:, 0] > 7.5

    assert weights[is_tail].mean() > weights[~is_tail].mean()


def test_multitask_missing_entries_produce_finite_weights():
    """Sparse multitask targets with NaN entries yield finite, positive weights."""
    targets = np.full((6, 2), np.nan)
    targets[:, 0] = [5.0, 5.0, 5.0, 8.0, np.nan, 5.0]
    targets[:4, 1] = [3.0, 3.0, 9.0, 3.0]

    weights = _inverse_density_weights(targets)

    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)
    assert weights.mean() == pytest.approx(1.0)


def test_weights_attached_only_to_training_loader():
    """Density weights are applied for train=True and left at unity otherwise."""
    smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CCCl", "CCBr"]
    targets = np.array([5.0, 5.0, 5.0, 8.0, 5.0, 5.0])
    featurizer = ChemPropFeaturizer(inverse_density_weighting=True, n_jobs=0)

    _, _, _, train_dataset = featurizer.featurize(smiles, targets, train=True)
    _, _, _, val_dataset = featurizer.featurize(smiles, targets, train=False)

    train_weights = [point.weight for point in train_dataset.data]
    val_weights = [point.weight for point in val_dataset.data]

    assert len(set(np.round(train_weights, 6))) > 1
    assert all(weight == pytest.approx(1.0) for weight in val_weights)


def test_disabled_flag_leaves_unit_weights():
    """With the flag off, every training datapoint keeps unit weight."""
    smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CCCl", "CCBr"]
    targets = np.array([5.0, 5.0, 5.0, 8.0, 5.0, 5.0])
    featurizer = ChemPropFeaturizer(inverse_density_weighting=False, n_jobs=0)

    _, _, _, dataset = featurizer.featurize(smiles, targets, train=True)

    assert all(point.weight == pytest.approx(1.0) for point in dataset.data)
