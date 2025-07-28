import numpy as np
import pytest
from numpy.testing import assert_array_equal

from openadmet.models.features.pairwise import PairFeaturizedData
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer


@pytest.fixture()
def smiles():
    return ["CCO", "CCN", "CCO"]

def test_pairwise_featurization(smiles):
    fp_featurizer = FingerprintFeaturizer(fp_type="ecfp")
    
    # Featurize the SMILES strings
    X, _ = fp_featurizer.featurize(smiles)

    paired_featurizer = PairFeaturizedData(how_to_pair="all")
    x_feat, y_paired, inds = paired_featurizer.pair_data(X, np.array([1.0, 2.0, 3.0]))

    assert x_feat.shape[0] == 9  # 3 choose 2 pairs + self-pairs
    assert x_feat.shape[1] == X.shape[1] * 2  # Each pair has two feature vectors concatenated
    assert y_paired.shape[0] == 9  # Same number of pairs as features
    assert len(inds) == 9   # Indices should match the number of pairs
    assert list(y_paired) == [0.0, -1.0, -2.0, 1.0, 0.0, -1.0, 2.0, 1.0, 0.0]  # Differences in y values

def test_pairwise_invalid_name():
    with pytest.raises(ValueError):
        PairFeaturizedData(how_to_pair="invalid_method")

def test_pairwise_rand(smiles):
    paired_featurizer = PairFeaturizedData(how_to_pair="rand", num_pairs=2)
    
    # Featurize the SMILES strings
    fp_featurizer = FingerprintFeaturizer(fp_type="ecfp")
    X, _ = fp_featurizer.featurize(smiles)

    x_feat, y_paired, inds = paired_featurizer.pair_data(X, np.array([1.0, 2.0, 3.0]))

    assert x_feat.shape[0] == 2  # Randomly sampled pairs
    assert y_paired.shape[0] == 2
    assert len(inds) == 2