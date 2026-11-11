import numpy as np
import pytest
from numpy.testing import assert_array_equal

from openadmet.models.features.combine import FeatureConcatenator
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.features.molfeat_properties import DescriptorFeaturizer
from openadmet.models.features.pairwise import PairwiseFeaturizer


def test_pairwise_make_new():
    """
    Verify that PairwiseFeaturizer can create a new independent instance via make_new().

    This is important for factory-like creation patterns in the registry or during cross-validation
    where fresh featurizers are needed.
    """
    featurizer = PairwiseFeaturizer(
        how_to_pair="full", featurizer=FingerprintFeaturizer(fp_type="ecfp:4")
    )
    new_featurizer = featurizer.make_new()
