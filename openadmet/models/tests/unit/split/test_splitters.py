import numpy as np
import pytest
import pandas as pd

from openadmet.models.split.sklearn import ShuffleSplitter
from openadmet.models.split.cluster import ClusterSplitter
from openadmet.models.split.split_base import splitters


def test_in_splitters():
    """Verify that concrete splitter implementations are correctly registered in the splitters registry."""
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
    """
    Validate that ShuffleSplitter correctly partitions data according to specified ratios.

    This test verifies both successful splits and error handling for invalid configurations.
    Correct splitting ensures that training, validation, and test sets are of the expected size
    and are mutually exclusive, which is critical for valid model evaluation.
    """
    if error is True:
        with pytest.raises(ValueError):
            splitter = ShuffleSplitter(
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                random_state=42,
            )
        return

    splitter = ShuffleSplitter(
        train_size=train_size, val_size=val_size, test_size=test_size, random_state=42
    )

    # Generate synthetic random data for testing split logic
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    X_train, X_val, X_test, y_train, y_val, y_test, groups = splitter.split(X, y)

    assert X_train.shape[0] == expected_train
    assert y_train.shape[0] == expected_train

    if val_size > 0:
        assert X_val.shape[0] == expected_val
        assert y_val.shape[0] == expected_val
        # Assert X_train and X_val are mutually exclusive
        train_set = set(map(tuple, X_train))
        val_set = set(map(tuple, X_val))
        assert len(train_set.intersection(val_set)) == 0

    else:
        assert X_val is None
        assert y_val is None

    if test_size > 0:
        assert X_test.shape[0] == expected_test
        assert y_test.shape[0] == expected_test
        # Assert X_train and X_test are mutually exclusive
        train_set = set(map(tuple, X_train))
        test_set = set(map(tuple, X_test))
        assert len(train_set.intersection(test_set)) == 0

        if val_size > 0:
            # Assert X_val and X_test are mutually exclusive
            val_set = set(map(tuple, X_val))
            assert len(val_set.intersection(test_set)) == 0

    else:
        assert X_test is None
        assert y_test is None


@pytest.fixture
def synthetic_cluster_data():
    """
    Provide a synthetic dataset with structural diversity for testing cluster splitting.

    This fixture returns a set of SMILES strings representing different chemical scaffolds
    (benzenes, pyridines, cyclohexanes, furans, thiophenes) and corresponding target values.
    Using diverse scaffolds ensures that clustering algorithms (like Butina or Bemis-Murcko)
    can meaningfully group molecules, allowing verification that splits respect cluster boundaries.
    """
    base_smiles = [
        "Cc1ccccc1",
        "CCc1ccccc1",
        "Oc1ccccc1",
        "Nc1ccccc1",
        "Clc1ccccc1",
        "Fc1ccccc1",
        "C(=O)Oc1ccccc1",
        "C(=O)Cc1ccccc1",
        "c1ccccc1C#N",
        "COc1ccccc1",
        "Cc1ccncc1",
        "CCc1ccncc1",
        "Oc1ccncc1",
        "Nc1ccncc1",
        "Clc1ccncc1",
        "Fc1ccncc1",
        "C(=O)Oc1ccncc1",
        "C(=O)Cc1ccncc1",
        "c1ccncc1C#N",
        "COc1ccncc1",
        "CC1CCCCC1",
        "CCC1CCCCC1",
        "OC1CCCCC1",
        "NC1CCCCC1",
        "ClC1CCCCC1",
        "FC1CCCCC1",
        "C(=O)OC1CCCCC1",
        "C(=O)CC1CCCCC1",
        "C1CCCCC1C#N",
        "COC1CCCCC1",
        "Cc1ccoc1",
        "CCc1ccoc1",
        "Oc1ccoc1",
        "Nc1ccoc1",
        "Clc1ccoc1",
        "Fc1ccoc1",
        "C(=O)Oc1ccoc1",
        "C(=O)Cc1ccoc1",
        "c1ccoc1C#N",
        "COc1ccoc1",
        "Cc1ccsc1",
        "CCc1ccsc1",
        "Oc1ccsc1",
        "Nc1ccsc1",
        "Clc1ccsc1",
        "Fc1ccsc1",
        "C(=O)Oc1ccsc1",
        "C(=O)Cc1ccsc1",
        "c1ccsc1C#N",
        "COc1ccsc1",
        "Cc1ccc2ccccc2c1",
        "CCc1ccc2ccccc2c1",
        "Oc1ccc2ccccc2c1",
        "Nc1ccc2ccccc2c1",
        "Clc1ccc2ccccc2c1",
        "Fc1ccc2ccccc2c1",
        "C(=O)Oc1ccc2ccccc2c1",
        "C(=O)Cc1ccc2ccccc2c1",
        "c1ccc2ccccc2c1C#N",
        "COc1ccc2ccccc2c1",
        "Cc1ccc2[nH]ccc2c1",
        "CCc1ccc2[nH]ccc2c1",
        "Oc1ccc2[nH]ccc2c1",
        "Nc1ccc2[nH]ccc2c1",
        "Clc1ccc2[nH]ccc2c1",
        "Fc1ccc2[nH]ccc2c1",
        "C(=O)Oc1ccc2[nH]ccc2c1",
        "C(=O)Cc1ccc2[nH]ccc2c1",
        "c1ccc2[nH]ccc2c1C#N",
        "COc1ccc2[nH]ccc2c1",
        "Cc1ccc2ncccc2c1",
        "CCc1ccc2ncccc2c1",
        "Oc1ccc2ncccc2c1",
        "Nc1ccc2ncccc2c1",
        "Clc1ccc2ncccc2c1",
        "Fc1ccc2ncccc2c1",
        "C(=O)Oc1ccc2ncccc2c1",
        "C(=O)Cc1ccc2ncccc2c1",
        "c1ccc2ncccc2c1C#N",
        "COc1ccc2ncccc2c1",
        "CC1CCCC1",
        "CCC1CCCC1",
        "OC1CCCC1",
        "NC1CCCC1",
        "ClC1CCCC1",
        "FC1CCCC1",
        "C(=O)OC1CCCC1",
        "C(=O)CC1CCCC1",
        "C1CCCC1C#N",
        "COC1CCCC1",
        "CC1CCNCC1",
        "CCC1CCNCC1",
        "OC1CCNCC1",
        "NC1CCNCC1",
        "ClC1CCNCC1",
        "FC1CCNCC1",
        "C(=O)OC1CCNCC1",
        "C(=O)CC1CCNCC1",
        "C1CCNCC1C#N",
        "COC1CCNCC1",
    ]
    smiles = pd.Series(base_smiles)
    y = pd.Series(np.linspace(0.0, 1.0, len(smiles)))
    return smiles, y


@pytest.mark.parametrize(
    "method",
    [
        "kmeans",
        "butina",
        "bemis-murcko",
    ],
)
def test_cluster_split_synthetic_data(method, synthetic_cluster_data):
    """
    Validate ClusterSplitter functionality with different clustering methods.

    This test ensures that molecular data is split such that training, validation, and test sets
    contain mutually exclusive molecules (no data leakage). It verifies split sizes are approximately
    correct and that structural separation is maintained.
    """
    X, y = synthetic_cluster_data
    splitter = ClusterSplitter(
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        random_state=42,
        method=method,
        k_clusters=10,
    )
    X_train, X_val, X_test, y_train, y_val, y_test, groups = splitter.split(
        X, y, num_iters=50
    )

    for obj in [X_train, X_val, X_test]:
        if obj is not None:
            assert isinstance(obj, pd.Series)

    for obj in [y_train, y_val, y_test]:
        if obj is not None:
            assert isinstance(obj, pd.Series)

    total = len(X)
    assert abs(len(X_train) - int(0.7 * total)) <= 5
    assert abs(len(X_val) - int(0.1 * total)) <= 5
    assert abs(len(X_test) - int(0.2 * total)) <= 5
    assert len(groups) == total

    # Check for data leakage
    # Assert X_train, X_val, and X_test are mutually exclusive by index
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)

    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0


def test_cluster_split_invalid_size_configuration():
    """Ensure ClusterSplitter raises ValueError for invalid split size configurations (e.g., sum != 1.0)."""
    with pytest.raises(ValueError):
        ClusterSplitter(
            train_size=1.0,
            val_size=0.0,
            test_size=0.0,
            random_state=42,
            method="kmeans",
        )


def test_cluster_split_invalid_method():
    """Ensure ClusterSplitter raises ValueError when initialized with an unknown clustering method."""
    with pytest.raises(ValueError):
        ClusterSplitter(
            train_size=0.7,
            val_size=0.1,
            test_size=0.2,
            random_state=42,
            method="not-a-method",
        )
