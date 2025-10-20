import pytest
from numpy.testing import assert_allclose

from openadmet.models.architecture.nepare import NeuralPairwiseRegressorModel


@pytest.fixture
def X_y():
    X = [[1, 2, 3], [4, 5, 6]]
    y = [1, 2]
    return X, y


def test_nepare():
    nepare_model = NeuralPairwiseRegressorModel()
    assert nepare_model.type == "NeuralPairwiseRegressorModel"
