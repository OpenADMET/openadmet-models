import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from openadmet.models.features.chemprop import ChemPropFeaturizer

_SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "CCCl", "CCBr"]
# two rows below the detection limit of 4.0, the rest above
_TARGETS = np.array([2.5, 5.0, 3.0, 8.0, 6.0, 5.5])


def test_subthreshold_rows_flagged_and_clamped_on_training_loader():
    """Training rows below the limit are flagged lt_mask and clamped up to the bound."""
    featurizer = ChemPropFeaturizer(left_censor_threshold=4.0, n_jobs=0)

    _, _, _, dataset = featurizer.featurize(_SMILES, _TARGETS, train=True)

    flagged = [bool(np.any(point.lt_mask)) for point in dataset.data]
    targets = [point.y[0] for point in dataset.data]

    assert flagged == [True, False, True, False, False, False]
    # the two sub-4 rows are clamped to the bound; the rest keep their exact value
    assert targets == pytest.approx([4.0, 5.0, 4.0, 8.0, 6.0, 5.5])


def test_above_threshold_rows_carry_no_censor_flag():
    """A row at or above the limit is never flagged and keeps its exact target."""
    featurizer = ChemPropFeaturizer(left_censor_threshold=4.0, n_jobs=0)

    _, _, _, dataset = featurizer.featurize(_SMILES, _TARGETS, train=True)

    above = dataset.data[3]  # target 8.0

    assert not np.any(above.lt_mask)
    assert above.y[0] == pytest.approx(8.0)


def test_censoring_applied_only_to_training_loader():
    """Validation and inference keep exact, unclamped targets with no censor flag."""
    featurizer = ChemPropFeaturizer(left_censor_threshold=4.0, n_jobs=0)

    _, _, _, val_dataset = featurizer.featurize(_SMILES, _TARGETS, train=False)

    flagged = [
        point.lt_mask is not None and bool(np.any(point.lt_mask))
        for point in val_dataset.data
    ]
    targets = [point.y[0] for point in val_dataset.data]

    assert not any(flagged)
    assert targets == pytest.approx(_TARGETS.tolist())


def test_disabled_threshold_leaves_targets_exact():
    """With no threshold set, every training row keeps its exact target and no flag."""
    featurizer = ChemPropFeaturizer(left_censor_threshold=None, n_jobs=0)

    _, _, _, dataset = featurizer.featurize(_SMILES, _TARGETS, train=True)

    flagged = [
        point.lt_mask is not None and bool(np.any(point.lt_mask))
        for point in dataset.data
    ]
    targets = [point.y[0] for point in dataset.data]

    assert not any(flagged)
    assert targets == pytest.approx(_TARGETS.tolist())


def test_scaler_fit_on_unclamped_targets():
    """The target scaler reflects the true distribution, not the clamped one, so the censored
    arm normalizes identically to an uncensored run and the effect is the loss, not a rescale."""
    featurizer = ChemPropFeaturizer(
        left_censor_threshold=4.0, normalize_targets=True, n_jobs=0
    )

    _, _, scaler, _ = featurizer.featurize(_SMILES, _TARGETS, train=True)

    expected = StandardScaler().fit(_TARGETS.reshape(-1, 1))
    assert scaler.mean_ == pytest.approx(expected.mean_)
    assert scaler.scale_ == pytest.approx(expected.scale_)
