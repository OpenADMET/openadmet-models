import numpy as np

from openadmet.models.split.split_base import splitters
from openadmet.models.split.sklearn import ShuffleSplitter


def test_in_splitters():
    assert "ShuffleSplitter" in splitters


def test_simple_split():
    splitter = ShuffleSplitter(train_size=0.8, test_size=0.2, random_state=42)
    X = np.random.rand(100, 10)
    Y = np.random.rand(100)
    X_train, X_test, y_train, y_test = splitter.split(X, Y)
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert y_train.shape[0] == 80
    assert y_test.shape[0] == 20
