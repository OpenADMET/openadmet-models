import pytest
import numpy as np
from numpy.testing import assert_array_equal


from openadmet_models.features.molfeat_properties import DescriptorFeaturizer
from openadmet_models.features.molfeat_fingerprint import FingerprintFeaturizer


@pytest.fixture()
def smiles():
    return ["CCO", "CCN", "CCO"]

@pytest.fixture()
def one_invalid_smi():
    return ["CCO", "CCN", "invalid", "CCO"]




@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("descr_type, shape", [("mordred", 1613), ("desc2d",215)])
def test_descriptor_featurizer(descr_type, shape, dtype):
    featurizer = DescriptorFeaturizer(descr_type=descr_type, dtype=dtype)
    X, idx = featurizer.featurize(["CCO", "CCN", "CCO"])
    assert X.shape == (3, shape)
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
    assert X.shape == (3, 1, 2000)
    assert X.dtype == dtype
    assert_array_equal(idx, np.arange(3))


def test_fingerprint_one_invalid(one_invalid_smi):
    featurizer = FingerprintFeaturizer(fp_type="ecfp")
    X, idx = featurizer.featurize(one_invalid_smi)
    assert X.shape == (3, 1, 2000)
    # index 2 is invalid, so the shape should be 3
    assert_array_equal(idx, np.asarray([0, 1, 3]))





