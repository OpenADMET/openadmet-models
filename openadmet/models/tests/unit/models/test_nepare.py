import pytest
from numpy.testing import assert_allclose

from openadmet.models.architecture.nepare import NeuralPairwiseRegressorModel


@pytest.fixture
def X_y():
    """Provide synthetic data pairs for testing pairwise regression."""
    X = [[1, 2, 3], [4, 5, 6]]
    y = [1, 2]
    return X, y


def test_nepare():
    """Verify initialization of the NeuralPairwiseRegressorModel."""
    nepare_model = NeuralPairwiseRegressorModel()
    assert nepare_model.type == "NeuralPairwiseRegressorModel"
