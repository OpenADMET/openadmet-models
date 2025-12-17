import numpy as np
import pytest
import pandas as pd

from openadmet.models.split.sklearn import ShuffleSplitter
from openadmet.models.split.cluster import ClusterSplitter
from openadmet.models.split.split_base import splitters
from openadmet.models.tests.unit.datafiles import CYP3A4_chembl_pchembl


def test_in_splitters():
    assert "ShuffleSplitter" in splitters
    assert "ClusterSplitter" in splitters


@pytest.mark.parametrize(
    "train_size, val_size, test_size, expected_train, expected_val, expected_test, error",
    # Train, val, test sizes
    [
        # 80, 0, 20
        (0.8, 0.0, 0.2, 80, 0, 20, False),
        # 70, 30, 0
        (0.7, 0.3, 0.0, 70, 30, 0, False),
        # 70, 10, 20
        (0.7, 0.1, 0.2, 70, 10, 20, False),
        # 60, 20, 20
        (0.6, 0.2, 0.2, 60, 20, 20, False),
        # 100, 0, 0; raises error
        (1.0, 0.0, 0.0, 100, 0, 0, False),
        # 50, 50, 50; raises error
        (0.5, 0.5, 0.5, -1, -1, -1, True),
    ],
)
def test_simple_split(
    train_size, val_size, test_size, expected_train, expected_val, expected_test, error
):
    # Error expected
    if error is True:
        with pytest.raises(ValueError):
            # Initialize splitter
            splitter = ShuffleSplitter(
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                random_state=42,
            )
        return

    # Initialize splitter
    splitter = ShuffleSplitter(
        train_size=train_size, val_size=val_size, test_size=test_size, random_state=42
    )

    # Generate random data
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    # Error is expected
    if error is True:
        with pytest.raises(ValueError):
            splitter.split(X, y)
        return

    # Perform the split
    X_train, X_val, X_test, y_train, y_val, y_test, groups = splitter.split(X, y)

    # Check train
    assert X_train.shape[0] == expected_train
    assert y_train.shape[0] == expected_train

    # Validation set requested
    if val_size > 0:
        assert X_val.shape[0] == expected_val
        assert y_val.shape[0] == expected_val

    # Validation set not requested
    else:
        assert X_val is None
        assert y_val is None

    # Test set requested
    if test_size > 0:
        assert X_test.shape[0] == expected_test
        assert y_test.shape[0] == expected_test

    # Test set not requested
    else:
        assert X_test is None
        assert y_test is None


@pytest.mark.parametrize(
    "train_size, val_size, test_size, expected_train, expected_val, expected_test, error, method",
    [
        # Test cases for kmeans
        (0.8, 0.0, 0.2, 1600, 0, 400, False, "kmeans"),
        (0.7, 0.3, 0.0, 1400, 600, 0, False, "kmeans"),
        (0.7, 0.1, 0.2, 1400, 200, 400, False, "kmeans"),
        (0.6, 0.2, 0.2, 1200, 400, 400, False, "kmeans"),
        # Test cases for butina
        (0.8, 0.0, 0.2, 1600, 0, 400, False, "butina"),
        (0.7, 0.3, 0.0, 1400, 600, 0, False, "butina"),
        (0.7, 0.1, 0.2, 1400, 200, 400, False, "butina"),
        (0.6, 0.2, 0.2, 1200, 400, 400, False, "butina"),
        # Test cases for bemis-murcko
        (0.8, 0.0, 0.2, 1600, 0, 400, False, "bemis-murcko"),
        (0.7, 0.3, 0.0, 1400, 600, 0, False, "bemis-murcko"),
        (0.7, 0.1, 0.2, 1400, 200, 400, False, "bemis-murcko"),
        (0.6, 0.2, 0.2, 1200, 400, 400, False, "bemis-murcko"),
        # Error cases
        (1.0, 0.0, 0.0, 200, 0, 0, True, "kmeans"),
        (0.5, 0.5, 0.5, -1, -1, -1, True, "kmeans"),
    ],
)
def test_cluster_split(
    train_size,
    val_size,
    test_size,
    expected_train,
    expected_val,
    expected_test,
    error,
    method,
):
    df = pd.read_csv(CYP3A4_chembl_pchembl)
    X = df["CANONICAL_SMILES"].values[:2000]
    y = df["pChEMBL mean"].values[:2000]

    # Error expected
    if error is True:
        with pytest.raises(ValueError):
            # Initialize splitter
            splitter = ClusterSplitter(
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                random_state=42,
                method=method,
                k_clusters=100,
            )
        return

    # Initialize splitter
    splitter = ClusterSplitter(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=42,
        method=method,
        k_clusters=100,
    )

    # Perform the split
    X_train, X_val, X_test, y_train, y_val, y_test, groups = splitter.split(X, y)

    # Check train
    assert abs(X_train.shape[0] - expected_train) <= 10
    assert abs(y_train.shape[0] - expected_train) <= 10

    # Validation set requested
    if val_size > 0:
        assert abs(X_val.shape[0] - expected_val) <= 10
        assert abs(y_val.shape[0] - expected_val) <= 10

    # Validation set not requested
    else:
        assert X_val is None
        assert y_val is None

    # Test set requested
    if test_size > 0:
        assert abs(X_test.shape[0] - expected_test) <= 10
        assert abs(y_test.shape[0] - expected_test) <= 10

    # Test set not requested
    else:
        assert X_test is None
        assert y_test is None
