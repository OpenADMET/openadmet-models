"""Tests for ImputeTransform."""

import numpy as np
import pytest
from pydantic import ValidationError

from openadmet.models.transforms.impute import ImputeTransform


@pytest.fixture
def features_with_gaps():
    """Provide a small matrix with one missing value per column."""
    return np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [6.0, 60.0],
            [np.nan, np.nan],
        ]
    )


def test_fit_returns_self_so_calls_chain(features_with_gaps):
    """fit must return the instance, which is what makes fit(X).transform(X) work."""
    transform = ImputeTransform(strategy="mean")
    assert transform.fit(features_with_gaps) is transform


def test_mean_strategy_fills_with_the_column_mean(features_with_gaps):
    """A missing entry must be replaced by the mean of the observed rows in its column."""
    out = (
        ImputeTransform(strategy="mean")
        .fit(features_with_gaps)
        .transform(features_with_gaps)
    )

    # Observed values are 1, 2, 6 and 10, 20, 60, so the means are 3 and 30
    np.testing.assert_allclose(out[3], np.array([3.0, 30.0]))
    np.testing.assert_allclose(out[:3], features_with_gaps[:3])


def test_median_strategy_fills_with_the_column_median(features_with_gaps):
    """The median strategy must use the middle observed value, not the mean."""
    out = (
        ImputeTransform(strategy="median")
        .fit(features_with_gaps)
        .transform(features_with_gaps)
    )

    # Medians of 1, 2, 6 and 10, 20, 60 are 2 and 20
    np.testing.assert_allclose(out[3], np.array([2.0, 20.0]))


def test_statistics_come_from_fit_rows_only(features_with_gaps):
    """Transforming a later batch must reuse the fitted statistics, not recompute them."""
    transform = ImputeTransform(strategy="mean").fit(features_with_gaps)

    # A batch whose own observed mean is 100, far from the fitted mean of 3
    later = np.array([[100.0, 1000.0], [np.nan, np.nan]])
    out = transform.transform(later)

    np.testing.assert_allclose(out[1], np.array([3.0, 30.0]))


def test_transform_before_fit_raises(features_with_gaps):
    """Using an unfitted transform must raise rather than silently pass data through."""
    with pytest.raises(RuntimeError, match="not been fitted"):
        ImputeTransform(strategy="mean").transform(features_with_gaps)


def test_iterative_imputer_is_reproducible_under_a_seed(features_with_gaps):
    """Two iterative imputers sharing a seed must produce identical output."""
    first = ImputeTransform(imputer="iterative", random_seed=42).fit(features_with_gaps)
    second = ImputeTransform(imputer="iterative", random_seed=42).fit(
        features_with_gaps
    )

    np.testing.assert_allclose(
        first.transform(features_with_gaps), second.transform(features_with_gaps)
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"strategy": "average"}, "Strategy must be one of", id="strategy"),
        pytest.param({"imputer": "knn"}, "Input should be", id="imputer"),
    ],
)
def test_rejects_unknown_configuration(kwargs, match):
    """An unsupported strategy or imputer must fail at construction."""
    with pytest.raises(ValidationError, match=match):
        ImputeTransform(**kwargs)
