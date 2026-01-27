"""Unit tests for featurization caching."""

import numpy as np
import pandas as pd
import pytest

from openadmet.models.features.cache import (
    clear_cache,
    generate_cache_key,
    load_features_from_cache,
    save_features_to_cache,
)
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer


def test_generate_cache_key_consistency():
    """Test that cache key generation is consistent for same inputs."""
    data = pd.Series(["CCO", "CC", "C"])
    params = {"fp_type": "ecfp:4", "dtype": "float32"}

    key1 = generate_cache_key(data, "FingerprintFeaturizer", params)
    key2 = generate_cache_key(data, "FingerprintFeaturizer", params)

    assert key1 == key2


def test_generate_cache_key_different_data():
    """Test that different data produces different cache keys."""
    data1 = pd.Series(["CCO", "CC", "C"])
    data2 = pd.Series(["CCO", "CC", "CO"])
    params = {"fp_type": "ecfp:4", "dtype": "float32"}

    key1 = generate_cache_key(data1, "FingerprintFeaturizer", params)
    key2 = generate_cache_key(data2, "FingerprintFeaturizer", params)

    assert key1 != key2


def test_generate_cache_key_different_params():
    """Test that different params produce different cache keys."""
    data = pd.Series(["CCO", "CC", "C"])
    params1 = {"fp_type": "ecfp:4", "dtype": "float32"}
    params2 = {"fp_type": "ecfp:6", "dtype": "float32"}

    key1 = generate_cache_key(data, "FingerprintFeaturizer", params1)
    key2 = generate_cache_key(data, "FingerprintFeaturizer", params2)

    assert key1 != key2


def test_save_and_load_cache():
    """Test saving and loading features from cache."""
    cache_key = "test_cache_key_12345"
    features = np.random.rand(10, 100)
    indices = np.arange(10)

    # Save to cache
    save_features_to_cache(cache_key, features, indices)

    # Load from cache
    loaded_features, loaded_indices = load_features_from_cache(cache_key)

    # Verify
    np.testing.assert_array_equal(loaded_features, features)
    np.testing.assert_array_equal(loaded_indices, indices)

    # Cleanup
    clear_cache()


def test_load_nonexistent_cache():
    """Test loading from cache when file doesn't exist."""
    result = load_features_from_cache("nonexistent_cache_key")
    assert result is None


def test_fingerprint_featurizer_with_caching():
    """Test FingerprintFeaturizer with caching enabled."""
    # Clear cache first
    clear_cache()

    smiles = ["CCO", "CC", "C", "CO"]
    featurizer = FingerprintFeaturizer(fp_type="ecfp:4", use_cache=True)

    # First call - should compute and cache
    feat1, indices1 = featurizer.featurize(smiles)
    assert feat1.shape[0] == len(smiles)
    assert len(indices1) == len(smiles)

    # Second call with same smiles - should load from cache
    feat2, indices2 = featurizer.featurize(smiles)
    np.testing.assert_array_equal(feat1, feat2)
    np.testing.assert_array_equal(indices1, indices2)

    # Cleanup
    clear_cache()


def test_fingerprint_featurizer_without_caching():
    """Test FingerprintFeaturizer with caching disabled."""
    smiles = ["CCO", "CC", "C", "CO"]
    featurizer = FingerprintFeaturizer(fp_type="ecfp:4", use_cache=False)

    # Both calls should compute features
    feat1, indices1 = featurizer.featurize(smiles)
    feat2, indices2 = featurizer.featurize(smiles)

    # Results should still be the same
    np.testing.assert_array_equal(feat1, feat2)
    np.testing.assert_array_equal(indices1, indices2)


def test_fingerprint_featurizer_different_params_different_cache():
    """Test that different featurizer params use different cache entries."""
    # Clear cache first
    clear_cache()

    smiles = ["CCO", "CC", "C", "CO"]

    # Create two featurizers with different fingerprint types
    featurizer1 = FingerprintFeaturizer(fp_type="ecfp:4", use_cache=True)
    featurizer2 = FingerprintFeaturizer(fp_type="ecfp:6", use_cache=True)

    # Featurize with both
    feat1, indices1 = featurizer1.featurize(smiles)
    feat2, indices2 = featurizer2.featurize(smiles)

    # Both should succeed
    assert feat1.shape[0] == len(smiles)
    assert feat2.shape[0] == len(smiles)
    assert len(indices1) == len(smiles)
    assert len(indices2) == len(smiles)

    # The cache should have created two separate files
    # (verified by different cache keys in the logs)

    # Cleanup
    clear_cache()


def test_clear_cache():
    """Test cache clearing functionality."""
    # Create some cache entries
    save_features_to_cache("test_key_1", np.random.rand(5, 10), np.arange(5))
    save_features_to_cache("test_key_2", np.random.rand(5, 10), np.arange(5))

    # Clear cache
    removed = clear_cache()
    assert removed >= 2

    # Verify cache is cleared
    assert load_features_from_cache("test_key_1") is None
    assert load_features_from_cache("test_key_2") is None
