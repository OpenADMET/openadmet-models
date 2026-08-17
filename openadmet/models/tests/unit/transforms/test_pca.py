"""Unit tests for PCATransform."""

import joblib
import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from openadmet.models.transforms.pca import PCATransform


@pytest.fixture()
def train_features():
    """Provide a deterministic (60, 20) train feature matrix."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(60, 20))


def test_pca_int_transform_matches_reference(train_features):
    """A single PCA must match a hand-built sklearn PCA at a fixed seed."""
    dims, seed = 5, 42
    transform = PCATransform(n_components=dims, random_seed=seed)
    out = transform.fit(train_features).transform(train_features)

    reference = (
        PCA(n_components=dims, random_state=seed)
        .fit(train_features)
        .transform(train_features)
    )
    np.testing.assert_allclose(out, reference, rtol=1e-12)
    assert out.shape == (60, dims)


def test_pca_dict_transform_matches_reference(train_features):
    """Per-block PCAs must match independently built one-block-at-a-time PCAs in block order."""
    blocks = [("A", 8), ("B", 12)]
    transform = PCATransform(n_components={"A": 3, "B": 4}, random_seed=42)
    out = transform.fit(train_features, feature_blocks=blocks).transform(train_features)

    reference = np.concatenate(
        [
            PCA(n_components=3, random_state=42)
            .fit(train_features[:, :8])
            .transform(train_features[:, :8]),
            PCA(n_components=4, random_state=42)
            .fit(train_features[:, 8:])
            .transform(train_features[:, 8:]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(out, reference, rtol=1e-12)
    assert out.shape == (60, 7)


def test_pca_fit_sees_train_rows_only(train_features):
    """Fitted statistics must come from the train matrix only, not from rows seen at transform time."""
    blocks = [("A", 10), ("B", 10)]
    transform = PCATransform(n_components={"A": 3, "B": 4}, random_seed=42)
    transform.fit(train_features, feature_blocks=blocks)

    (_, start_a, stop_a, pipeline_a), (_, _, _, _) = transform._pca_blocks
    pca_a = pipeline_a.named_steps["pca"]
    np.testing.assert_allclose(
        pca_a.mean_, train_features[:, start_a:stop_a].mean(axis=0), rtol=1e-12
    )
    assert pca_a.n_features_in_ == stop_a - start_a


def test_pca_impute_strategy_matches_reference(train_features):
    """NaN handling must match a hand-built (SimpleImputer, PCA) pipeline at the same strategy and seed."""
    X = train_features[:30, :20].copy()
    X[0, 3] = np.nan
    X[5, 14] = np.nan
    blocks = [("A", 10), ("B", 10)]
    transform = PCATransform(
        n_components={"A": 2, "B": 3},
        impute_strategy="median",
        random_seed=42,
    )
    out = transform.fit(X, feature_blocks=blocks).transform(X)

    reference = np.concatenate(
        [
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("pca", PCA(n_components=2, random_state=42)),
                ]
            )
            .fit(X[:, :10])
            .transform(X[:, :10]),
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("pca", PCA(n_components=3, random_state=42)),
                ]
            )
            .fit(X[:, 10:])
            .transform(X[:, 10:]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(out, reference, rtol=1e-12)


def test_pca_transform_width_mismatch_raises(train_features):
    """Transform-time matrices must carry the fitted column layout in both directions."""
    transform = PCATransform(n_components=3).fit(train_features)

    with pytest.raises(ValueError, match="expects 20"):
        transform.transform(train_features[:, :18])

    wider = np.concatenate([train_features, train_features[:, :2]], axis=1)
    with pytest.raises(ValueError, match="expects 20"):
        transform.transform(wider)


def test_pca_transform_before_fit_raises(train_features):
    """Using an unfitted PCA must raise a RuntimeError, mirroring ImputeTransform."""
    transform = PCATransform(n_components=2)
    with pytest.raises(RuntimeError, match="not been fitted"):
        transform.transform(train_features)


@pytest.mark.parametrize(
    "value, match",
    [
        pytest.param(0, "n_components must be >= 1", id="zero"),
        pytest.param({}, "at least one entry", id="empty_dict"),
        pytest.param({"A": 0}, "int >= 1", id="non_positive_entry"),
    ],
)
def test_pca_rejects_invalid_n_components(value, match):
    """Construction must reject non-positive, empty, and boolean component counts."""
    with pytest.raises(ValidationError, match=match):
        PCATransform(n_components=value)


def test_pca_dict_without_blocks_raises(train_features):
    """Per-block PCA without feature_blocks must fail loudly instead of guessing."""
    transform = PCATransform(n_components={"A": 2})
    with pytest.raises(ValueError, match="requires feature_blocks"):
        transform.fit(train_features)


def test_pca_dict_key_mismatch_raises(train_features):
    """n_components keys must match the block keys exactly in both directions."""
    transform = PCATransform(n_components={"A": 2, "C": 3})
    with pytest.raises(ValueError, match="keys must exactly match"):
        transform.fit(train_features, feature_blocks=[("A", 10), ("B", 10)])

    transform_missing = PCATransform(n_components={"A": 2})
    with pytest.raises(ValueError, match="keys must exactly match"):
        transform_missing.fit(train_features, feature_blocks=[("A", 10), ("B", 10)])


def test_pca_duplicate_block_keys_raise(train_features):
    """Duplicate block keys are ambiguous and must be rejected."""
    transform = PCATransform(n_components={"A": 2, "B": 2})
    with pytest.raises(ValueError, match="Duplicate feature block keys"):
        transform.fit(train_features, feature_blocks=[("A", 10), ("A", 10)])


def test_pca_fit_width_mismatch_raises(train_features):
    """Blocks that do not cover the input width (e.g. behind a width-changing transform) must be rejected."""
    transform = PCATransform(n_components={"A": 2, "B": 2})
    with pytest.raises(ValueError, match="widths sum to 40 but the input has 20"):
        transform.fit(train_features, feature_blocks=[("A", 20), ("B", 20)])


def test_pca_dims_at_or_above_block_rank_raises(train_features):
    """Component counts must stay below min(train rows, block width)."""
    transform = PCATransform(n_components={"A": 10, "B": 2})
    with pytest.raises(ValueError, match="must be smaller than min"):
        transform.fit(train_features, feature_blocks=[("A", 10), ("B", 10)])

    transform_int = PCATransform(n_components=20)
    with pytest.raises(ValueError, match="must be smaller than min"):
        transform_int.fit(train_features)


def test_pca_transform_rejects_1d_input(train_features):
    """A 1D input must be rejected with a clear shape error."""
    transform = PCATransform(n_components=2)
    transform.fit(train_features)
    with pytest.raises(ValueError, match="2D feature matrix"):
        transform.transform(train_features[0])


def test_pca_joblib_roundtrip_preserves_fitted_state(train_features, tmp_path):
    """A persisted fitted transform must reproduce identical projections."""
    blocks = [("A", 10), ("B", 10)]
    transform = PCATransform(n_components={"A": 3, "B": 2}, random_seed=42)
    transform.fit(train_features, feature_blocks=blocks)
    expected = transform.transform(train_features)

    path = tmp_path / "transform.pickle"
    payload = {"schema": "v1", "transforms": [transform], "sklearn_version": "test"}
    with open(path, "wb") as f:
        joblib.dump(payload, f)
    with open(path, "rb") as f:
        loaded = joblib.load(f)["transforms"][0]

    np.testing.assert_allclose(loaded.transform(train_features), expected, rtol=1e-12)
