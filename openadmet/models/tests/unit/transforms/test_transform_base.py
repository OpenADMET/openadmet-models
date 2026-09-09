"""Tests for the transform sequence helpers in transform_base."""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from openadmet.models.transforms.impute import ImputeTransform
from openadmet.models.transforms.pca import PCATransform
from openadmet.models.transforms.transform_base import (
    fit_transforms,
    to_transform_list,
    transform_features,
)


def test_to_transform_list_normalizes_single_transform():
    """A bare transform instance must normalize to a one-element list."""
    t = ImputeTransform()
    assert to_transform_list(t) == [t]


def test_to_transform_list_passes_sequences_through():
    """Sequences must keep order and element identity."""
    a, b = ImputeTransform(), PCATransform(n_components=2)
    assert to_transform_list([a, b]) == [a, b]
    assert to_transform_list((a, b)) == [a, b]


def test_fit_transforms_threads_outputs_between_elements():
    """Each element must be fit on the previous element's output, not the raw input."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 10))
    X[0, 2] = np.nan
    X[3, 7] = np.nan

    imputer = ImputeTransform(strategy="mean")
    pca = PCATransform(n_components=3, random_seed=42)
    out = fit_transforms([imputer, pca], X)

    imputed = imputer.fit(X).transform(X)
    reference = pca.fit(imputed).transform(imputed)
    np.testing.assert_allclose(out, reference, rtol=1e-10)
    assert out.shape == (40, 3)


def test_fit_transforms_forwards_blocks_through_the_sequence():
    """feature_blocks must reach a per-block PCA sitting behind a layout-agnostic element."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 20))

    imputer = ImputeTransform(strategy="mean")
    pca = PCATransform(n_components={"A": 3, "B": 2}, random_seed=42)
    out = fit_transforms([imputer, pca], X, feature_blocks=[("A", 10), ("B", 10)])

    imputed = ImputeTransform(strategy="mean").fit(X).transform(X)
    reference = np.concatenate(
        [
            PCA(n_components=3, random_state=42)
            .fit(imputed[:, :10])
            .transform(imputed[:, :10]),
            PCA(n_components=2, random_state=42)
            .fit(imputed[:, 10:])
            .transform(imputed[:, 10:]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(out, reference, rtol=1e-10)
    assert out.shape == (40, 5)


def test_transform_features_applies_fitted_sequence_in_order():
    """transform_features applies elements in sequence without fitting."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 12))

    imputer = ImputeTransform(strategy="median").fit(X)
    pca = PCATransform(n_components=3, random_seed=42).fit(X)
    out = transform_features([imputer, pca], X)

    reference = pca.transform(imputer.transform(X))
    np.testing.assert_allclose(out, reference, rtol=1e-10)


def test_transform_features_raises_when_element_unfitted():
    """An unfitted element on the inference path must raise its own RuntimeError."""
    X = np.random.default_rng(4).normal(size=(5, 4))
    imputer = ImputeTransform(strategy="mean")  # never fitted
    with pytest.raises(RuntimeError, match="not been fitted"):
        transform_features(imputer, X)
