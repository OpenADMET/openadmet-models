"""Tests for CheMeleonEmbeddingFeaturizer."""

import numpy as np
import pytest

from openadmet.models.features.chemeleon_embedding import CheMeleonEmbeddingFeaturizer


@pytest.fixture
def smiles():
    return ["CCO", "CCN", "c1ccccc1"]


def test_featurize_shape_dtype_indices(smiles):
    featurizer = CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=2)
    embeddings, indices = featurizer.featurize(smiles)

    assert embeddings.shape == (3, 2048)
    assert embeddings.dtype == np.float32
    assert np.array_equal(indices, np.arange(3))


def test_featurize_batch_size_forwarding(smiles):
    featurizer = CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=2)
    embeddings, indices = featurizer.featurize(smiles)

    assert embeddings.shape[0] == len(smiles)
    assert len(indices) == len(smiles)


def test_featurizer_respects_accelerator():
    featurizer = CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=256)
    embeddings, _ = featurizer.featurize(["CCO", "CCN"])

    model = featurizer._ensure_model().estimator
    device = next(model.parameters()).device
    assert str(device) == "cpu"


def test_featurizer_compatible_with_concatenator(smiles):
    from openadmet.models.features.combine import FeatureConcatenator
    from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer

    concat = FeatureConcatenator(featurizers=[
        CheMeleonEmbeddingFeaturizer(accelerator="cpu", batch_size=2),
        FingerprintFeaturizer(fp_type="ecfp:4", n_jobs=1)
    ])

    X, idx = concat.featurize(smiles)
    # CheMeleon 2048 + ECFP 2000 = 4048
    assert X.shape == (3, 4048)
    assert np.array_equal(idx, np.arange(3))
