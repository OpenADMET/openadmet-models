import numpy as np
import pytest
from numpy.testing import assert_array_equal

from openadmet.models.features.combine import FeatureConcatenator
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.features.molfeat_properties import DescriptorFeaturizer


@pytest.fixture()
def smiles():
    return ["CCO", "CCN", "CCO"]


@pytest.fixture()
def one_invalid_smi():
    return ["CCO", "CCN", "invalid", "CCO"]


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("descr_type", ["mordred", "desc2d"])
def test_descriptor_featurizer(descr_type, dtype):
    featurizer = DescriptorFeaturizer(descr_type=descr_type, dtype=dtype)
    X, idx = featurizer.featurize(["CCO", "CCN", "CCO"])
    assert X.dtype == dtype
    assert_array_equal(idx, np.arange(3))


def test_descriptor_one_invalid(one_invalid_smi):
    featurizer = DescriptorFeaturizer(descr_type="mordred")
    X, idx = featurizer.featurize(one_invalid_smi)
    assert X.shape == (3, 1613)
    # index 2 is invalid, so the shape should be 3
    assert_array_equal(idx, np.asarray([0, 1, 3]))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("fp_type", ("ecfp", "fcfp"))
def test_fingerprint_featurizer(smiles, fp_type, dtype):
    featurizer = FingerprintFeaturizer(fp_type=fp_type, dtype=dtype)
    X, idx = featurizer.featurize(smiles)
    assert X.shape == (3, 2000)
    assert X.dtype == dtype
    assert_array_equal(idx, np.arange(3))


def test_fingerprint_one_invalid(one_invalid_smi):
    featurizer = FingerprintFeaturizer(fp_type="ecfp")
    X, idx = featurizer.featurize(one_invalid_smi)
    assert X.shape == (3, 2000)
    # index 2 is invalid, so the shape should be 3
    assert_array_equal(idx, np.asarray([0, 1, 3]))


def test_feature_concatenator(smiles):
    desc_featurizer = DescriptorFeaturizer(descr_type="mordred")
    fp_featurizer = FingerprintFeaturizer(fp_type="ecfp")
    concat = FeatureConcatenator(featurizers=[desc_featurizer, fp_featurizer])
    X, idx = concat.featurize(smiles)
    assert X.shape == (3, 3613)
    assert_array_equal(idx, np.arange(3))


def test_feature_concatenator_failed_diff_positions(one_invalid_smi):
    desc_featurizer = DescriptorFeaturizer(descr_type="mordred")
    fp_featurizer = FingerprintFeaturizer(fp_type="ecfp")
    concat = FeatureConcatenator(featurizers=[desc_featurizer, fp_featurizer])
    X, idx = concat.featurize(one_invalid_smi)
    assert X.shape == (3, 3613)
    # index 2 is invalid, so the shape should be 3
    assert_array_equal(idx, np.asarray([0, 1, 3]))
