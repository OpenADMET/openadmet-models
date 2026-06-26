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


def test_bandwidth_changes_the_resulting_weights(skewed_targets):
    """Different KDE bandwidths produce materially different per-sample weights."""
    narrow = _inverse_density_weights(skewed_targets, bandwidth=0.2)
    wide = _inverse_density_weights(skewed_targets, bandwidth=1.0)

    assert not np.allclose(narrow, wide)


def test_higher_clip_factor_raises_the_weight_ceiling(skewed_targets):
    """A larger clip factor lets rare-tail weights climb higher before clipping."""
    is_tail = skewed_targets[:, 0] > 7.5

    loose = _inverse_density_weights(skewed_targets, weight_clip_median_factor=20.0)
    tight = _inverse_density_weights(skewed_targets, weight_clip_median_factor=2.0)

    assert loose[is_tail].max() > tight[is_tail].max()


def test_featurizer_defaults_match_module_constants(skewed_targets):
    """The featurizer's default params reproduce the bare module-level weighting."""
    from openadmet.models.features.chemprop import (
        KDE_BANDWIDTH,
        WEIGHT_CLIP_MEDIAN_FACTOR,
    )

    featurizer = ChemPropFeaturizer(inverse_density_weighting=True, n_jobs=0)

    assert featurizer.kde_bandwidth == KDE_BANDWIDTH
    assert featurizer.weight_clip_median_factor == WEIGHT_CLIP_MEDIAN_FACTOR


def test_featurizer_passes_bandwidth_through_to_weights():
    """A featurizer bandwidth override reaches the weighting and changes the tail weight."""
    smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CCCl", "CCBr"]
    targets = np.array([5.0, 5.0, 5.0, 8.0, 5.0, 5.0])

    narrow = ChemPropFeaturizer(
        inverse_density_weighting=True, kde_bandwidth=0.2, n_jobs=0
    )
    wide = ChemPropFeaturizer(
        inverse_density_weighting=True, kde_bandwidth=2.0, n_jobs=0
    )

    _, _, _, narrow_ds = narrow.featurize(smiles, targets, train=True)
    _, _, _, wide_ds = wide.featurize(smiles, targets, train=True)

    narrow_tail = next(p.weight for p in narrow_ds.data if p.y[0] == 8.0)
    wide_tail = next(p.weight for p in wide_ds.data if p.y[0] == 8.0)

    assert narrow_tail != pytest.approx(wide_tail)
