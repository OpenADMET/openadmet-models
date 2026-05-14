import numpy as np
import pytest
from numpy.testing import assert_array_equal

from openadmet.models.features.combine import FeatureConcatenator
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.features.molfeat_properties import DescriptorFeaturizer
from openadmet.models.features.pairwise import PairwiseFeaturizer


@pytest.fixture()
def smiles():
    """Provide a list of valid SMILES strings for testing featurization."""
    return ["CCO", "CCN", "CCO"]


@pytest.fixture()
def one_invalid_smi():
    """Provide a list of SMILES strings containing one invalid entry to test error handling."""
    return ["CCO", "CCN", "invalid", "CCO"]


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("descr_type", ["mordred", "desc2d"])
def test_descriptor_featurizer(descr_type, dtype):
    """
    Validate DescriptorFeaturizer for different descriptor types and floating point precisions.

    This ensures that physical-chemical descriptors (like Mordred or RDKit 2D) are correctly generated
    and returned with the requested data type, which is important for downstream model compatibility.
    """
    featurizer = DescriptorFeaturizer(descr_type=descr_type, dtype=dtype)
    X, idx = featurizer.featurize(["CCO", "CCN", "CCO"])
    assert X.dtype == dtype
    assert_array_equal(idx, np.arange(3))


def test_descriptor_one_invalid(one_invalid_smi):
    """
    Ensure DescriptorFeaturizer robustly handles invalid SMILES strings.

    The featurizer should skip invalid molecules and return indices corresponding only to the valid ones.
    This prevents the entire pipeline from crashing due to a single bad input.
    """
    featurizer = DescriptorFeaturizer(descr_type="mordred")
    X, idx = featurizer.featurize(one_invalid_smi)
    assert X.shape == (3, 1613)
    # index 2 is invalid, so the shape should be 3
    assert_array_equal(idx, np.asarray([0, 1, 3]))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("fp_type", ("ecfp", "fcfp"))
def test_fingerprint_featurizer(smiles, fp_type, dtype):
    """
    Validate FingerprintFeaturizer for different fingerprint types (ECFP, FCFP) and precisions.

    This verifies that structural fingerprints are correctly generated with the expected vector size (2000)
    and data type.
    """
    featurizer = FingerprintFeaturizer(fp_type=fp_type, dtype=dtype)
    X, idx = featurizer.featurize(smiles)
    assert X.shape == (3, 2000)
    assert X.dtype == dtype
    assert_array_equal(idx, np.arange(3))


def test_fingerprint_one_invalid(one_invalid_smi):
    """
    Ensure FingerprintFeaturizer robustly handles invalid SMILES strings.

    Similar to descriptors, it should filter out invalid entries and return correct indices for valid ones.
    """
    featurizer = FingerprintFeaturizer(fp_type="ecfp")
    X, idx = featurizer.featurize(one_invalid_smi)
    assert X.shape == (3, 2000)
    # index 2 is invalid, so the shape should be 3
    assert_array_equal(idx, np.asarray([0, 1, 3]))


def test_feature_concatenator(smiles):
    """
    Validate that FeatureConcatenator correctly combines multiple feature sets (descriptors + fingerprints).

    This ensures that different feature representations can be stacked horizontally for the same molecules,
    providing a richer feature set for training.
    """
    desc_featurizer = DescriptorFeaturizer(descr_type="mordred")
    fp_featurizer = FingerprintFeaturizer(fp_type="ecfp")
    concat = FeatureConcatenator(featurizers=[desc_featurizer, fp_featurizer])
    X, idx = concat.featurize(smiles)
    assert X.shape == (3, 3613)
    assert_array_equal(idx, np.arange(3))


def test_feature_concatenator_drops_intersection(mocker):
    """
    Verify that FeatureConcatenator only keeps molecules valid across ALL featurizers.

    If one featurizer fails for molecule A and another fails for molecule B, the concatenator
    must drop both A and B to maintain feature alignment.

    We mock the underlying featurizers to control which indices fail, avoiding the need for
    complex real-world molecules that fail specific featurizers. This isolates the intersection logic.
    """
    # Arrange
    desc_featurizer = DescriptorFeaturizer(descr_type="mordred")
    fp_featurizer = FingerprintFeaturizer(fp_type="ecfp")
    concat = FeatureConcatenator(featurizers=[desc_featurizer, fp_featurizer])

    # Mock descriptor featurizer to return 3 valid outputs (fails on index 1)
    # Indices: 0, 2, 3 (skips 1)
    desc_features = np.zeros((3, 1613))
    desc_indices = np.array([0, 2, 3])
    mocker.patch.object(
        DescriptorFeaturizer, "featurize", return_value=(desc_features, desc_indices)
    )

    # Mock fingerprint featurizer to return 3 valid outputs (fails on index 2)
    # Indices: 0, 1, 3 (skips 2)
    # Note: ECFP size is 2000 in this codebase
    fp_features = np.zeros((3, 2000))
    fp_indices = np.array([0, 1, 3])
    mocker.patch.object(
        FingerprintFeaturizer, "featurize", return_value=(fp_features, fp_indices)
    )

    smiles = ["SMI0", "SMI1", "SMI2", "SMI3"]

    # Act
    X, idx = concat.featurize(smiles)

    # Assert
    # Intersection of [0, 2, 3] and [0, 1, 3] is [0, 3]
    # Expected shape: (2, 1613 + 2000) = (2, 3613)
    assert X.shape == (2, 3613)
    assert_array_equal(idx, np.array([0, 3]))


def test_feature_concatenator_order_independence(smiles):
    """
    Ensure that changing the order of featurizers in the list results in the same outcome due to sorting.
    """
    desc_featurizer = DescriptorFeaturizer(descr_type="mordred")
    fp_featurizer = FingerprintFeaturizer(fp_type="ecfp")

    concat1 = FeatureConcatenator(featurizers=[desc_featurizer, fp_featurizer])
    X1, idx1 = concat1.featurize(smiles)

    concat2 = FeatureConcatenator(featurizers=[fp_featurizer, desc_featurizer])
    X2, idx2 = concat2.featurize(smiles)

    assert_array_equal(X1, X2)
    assert_array_equal(idx1, idx2)


def test_pairwise_featurizer(smiles):
    """
    Validate PairwiseFeaturizer in 'full' mode (all-pairs).

    This tests that features are generated for every pair of molecules and that target values
    (differences) are correctly computed.
    """
    featurizer = PairwiseFeaturizer(
        featurizer={"FingerprintFeaturizer": {"fp_type": "ecfp", "dtype": np.float32}},
        how_to_pair="full",
        batch_size=2,
        shuffle=False,
        n_jobs=1,
    )
    _, _, scaler, dataset = featurizer.featurize(smiles, y=np.array([1.0, 2.0, 1.0]))
    assert len(dataset) == 9  # 3 molecules -> 3x3 = 9 pairs in 'full' mode
    expected_y = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    assert [dataset[i][2] for i in range(9)] == pytest.approx(expected_y)
    assert scaler is None  # No scaling applied in FingerprintFeaturizer
