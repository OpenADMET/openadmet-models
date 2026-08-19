"""Tests for CheMeleonEmbeddingFeaturizer."""

import numpy as np
import pytest

from openadmet.models.features import chemeleon_embedding
from openadmet.models.features.chemeleon_embedding import CheMeleonEmbeddingFeaturizer
from openadmet.models.features.combine import FeatureConcatenator
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer


@pytest.fixture(autouse=True)
def _hermetic_foundation(monkeypatch):
    """Run against the random-weight chemeleon-test architecture so no checkpoint is downloaded."""
    monkeypatch.setattr(chemeleon_embedding, "_FOUNDATION_NAME", "chemeleon-test")


@pytest.fixture
def smiles():
    return ["CCO", "CCN", "c1ccccc1"]


def test_featurize_shape_dtype_indices(smiles):
    featurizer = CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=2)
    embeddings, indices = featurizer.featurize(smiles)

    assert embeddings.shape == (3, 2048)
    assert embeddings.dtype == np.float32
    assert np.array_equal(indices, np.arange(3))


def test_featurize_batch_invariance(smiles):
    """Embeddings and indices must not depend on the chosen forward batch size."""
    featurizer = CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=1)
    emb_small, idx_small = featurizer.featurize(smiles)

    # One model, two forward batch sizes: only the batching may differ
    featurizer.batch_size = 256
    emb_large, idx_large = featurizer.featurize(smiles)

    # float32 reduction order varies with batch shape, so allow last-ulp drift
    np.testing.assert_allclose(emb_small, emb_large, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(idx_small, idx_large)


def test_featurizer_respects_accelerator():
    featurizer = CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=256)
    embeddings, _ = featurizer.featurize(["CCO", "CCN"])

    model = featurizer._ensure_model().estimator
    device = next(model.parameters()).device
    assert str(device) == "cpu"


def test_featurize_invalid_smiles_raises():
    """Unparseable SMILES propagate the toolkit error instead of being skipped."""
    featurizer = CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=2)
    with pytest.raises(RuntimeError, match="not_a_smile"):
        featurizer.featurize(["CCO", "not_a_smile", "CCN"])


def test_featurize_empty_input_returns_empty():
    """An empty input returns an empty matrix and indices without building the model."""
    featurizer = CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=2)
    embeddings, indices = featurizer.featurize([])

    assert embeddings.shape == (0, 2048)
    np.testing.assert_array_equal(indices, np.empty(0, dtype=int))
    assert featurizer._model is None


def test_featurizer_compatible_with_concatenator(smiles):
    concat = FeatureConcatenator(
        featurizers=[
            CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=2),
            FingerprintFeaturizer(fp_type="ecfp:4", n_jobs=1),
        ]
    )

    X, idx = concat.featurize(smiles)

    # CheMeleon 2048 + ECFP 2000 = 4048
    assert X.shape == (3, 4048)
    assert np.array_equal(idx, np.arange(3))
