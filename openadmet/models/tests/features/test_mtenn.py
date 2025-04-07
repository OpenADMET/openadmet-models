import numpy as np
import pytest
from numpy.testing import assert_array_equal
from openadmet.models.features.mtenn import MTENNFeaturizer, MTENNDataset
from openadmet.models.tests.test_data

def test_mtenn_dataset()