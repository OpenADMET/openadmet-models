"""Unit tests for FeatureConcatenator entry forms, duplicate rejection, and feature_blocks."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from openadmet.models.features.combine import FeatureConcatenator
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.features.molfeat_properties import DescriptorFeaturizer
from openadmet.models.features.null_featurizer import NullFeaturizer


@pytest.fixture()
def smiles():
    """Provide a list of valid SMILES strings for testing featurization."""
    return ["CCO", "CCN", "CCO", "CC(=O)OC", "c1ccccc1"]


@pytest.mark.parametrize(
    "entry, expected_name",
    [
        pytest.param(
            {"type": "FingerprintFeaturizer", "params": {"fp_type": "ecfp"}},
            "FingerprintFeaturizer",
            id="wrapper",
        ),
        pytest.param(
            {"DescriptorFeaturizer": {"descr_type": "desc2d"}},
            "DescriptorFeaturizer",
            id="single_key",
        ),
        pytest.param(NullFeaturizer(n_jobs=1), "NullFeaturizer", id="instance"),
    ],
)
def test_concatenator_list_entry_forms(entry, expected_name):
    """List forms must accept {type, params} wrappers, single-type entries, and instances."""
    concat = FeatureConcatenator(featurizers=[entry])
    assert [type(f).__name__ for f in concat.featurizers] == [expected_name]


def test_concatenator_rejects_duplicate_classes():
    """Same-class featurizers are ambiguous for per-key transforms and must be rejected."""
    with pytest.raises(
        ValidationError, match="cannot combine multiple featurizers of the same class"
    ):
        FeatureConcatenator(
            featurizers=[
                {"type": "FingerprintFeaturizer", "params": {"fp_type": "ecfp"}},
                {"type": "FingerprintFeaturizer", "params": {"fp_type": "ecfp:4"}},
            ]
        )


def test_concatenator_rejects_ambiguous_list_entry():
    """A multi-key dict that is not a {type, params} wrapper must be rejected."""
    with pytest.raises(ValidationError, match="must be"):
        FeatureConcatenator(
            featurizers=[
                {"FingerprintFeaturizer": {"fp_type": "ecfp"}, "extra": 1},
            ]
        )


def test_concatenator_dict_form_still_constructs():
    """The original dict form must keep working unchanged."""
    concat = FeatureConcatenator(
        featurizers={
            "FingerprintFeaturizer": {"fp_type": "ecfp", "n_jobs": 1},
            "DescriptorFeaturizer": {"descr_type": "desc2d", "n_jobs": 1},
        }
    )
    names = [type(f).__name__ for f in concat.featurizers]
    assert names == ["DescriptorFeaturizer", "FingerprintFeaturizer"]


def test_concatenator_feature_blocks_align_with_matrix(smiles):
    """feature_blocks must cover the concatenated matrix exactly, in the same order the columns are emitted."""
    concat = FeatureConcatenator(
        featurizers=[
            {
                "type": "FingerprintFeaturizer",
                "params": {"fp_type": "ecfp", "n_jobs": 1},
            },
            {
                "type": "DescriptorFeaturizer",
                "params": {"descr_type": "desc2d", "n_jobs": 1},
            },
        ]
    )
    blocks = concat.feature_blocks(smiles)
    feats, idx = concat.featurize(smiles)

    # every probe SMILES is valid here, so all rows survive
    assert_array_equal(idx, np.arange(len(smiles)))

    # stable class-name order puts the descriptor block first
    descr_feats, _ = concat.featurizers[0].featurize(smiles)
    fp_feats, _ = concat.featurizers[1].featurize(smiles)
    assert blocks == [
        ("DescriptorFeaturizer", descr_feats.shape[1]),
        ("FingerprintFeaturizer", fp_feats.shape[1]),
    ]
    assert feats.shape == (
        len(smiles),
        descr_feats.shape[1] + fp_feats.shape[1],
    )
    assert (feats[:, : descr_feats.shape[1]] == descr_feats).all()
    assert (feats[:, descr_feats.shape[1] :] == fp_feats).all()


def test_concatenator_nested_feature_blocks_flatten(smiles):
    """Nested FeatureConcatenator blocks must flatten into the outer block list in matrix order."""
    inner = [
        {"type": "FingerprintFeaturizer", "params": {"fp_type": "ecfp", "n_jobs": 1}},
        {
            "type": "DescriptorFeaturizer",
            "params": {"descr_type": "desc2d", "n_jobs": 1},
        },
    ]
    outer = FeatureConcatenator(
        featurizers=[
            {"type": "FeatureConcatenator", "params": {"featurizers": inner}},
            {"type": "NullFeaturizer", "params": {}},
        ]
    )
    blocks = outer.feature_blocks(smiles)
    feats, _ = outer.featurize(smiles)

    keys = [key for key, _ in blocks]
    assert keys == ["DescriptorFeaturizer", "FingerprintFeaturizer", "NullFeaturizer"]
    assert sum(width for _, width in blocks) == feats.shape[1]
    assert blocks[2] == ("NullFeaturizer", 1)


def test_feature_blocks_default_single_block_key(smiles):
    """A plain featurizer must report one block keyed by its registry name."""
    fp = FingerprintFeaturizer(fp_type="ecfp", n_jobs=1)
    blocks = fp.feature_blocks(smiles)
    feats, _ = fp.featurize(smiles)

    assert len(blocks) == 1
    assert blocks[0][0] == "FingerprintFeaturizer"
    assert blocks[0][1] == feats.shape[1]


def test_feature_blocks_probe_skips_invalid_entries():
    """The probe must try entries in order until one featurizes successfully."""
    fp = FingerprintFeaturizer(fp_type="ecfp", n_jobs=1)
    feats, _ = fp.featurize(["CCO"])
    width = feats.shape[1] if feats.ndim > 1 else np.atleast_2d(feats).shape[1]

    blocks = fp.feature_blocks(["invalid_smiles", "CCO"])
    assert blocks == [("FingerprintFeaturizer", width)]


def test_feature_blocks_probe_all_invalid_raises():
    """An all-invalid probe must raise a clear error."""
    fp = FingerprintFeaturizer(fp_type="ecfp", n_jobs=1)
    with pytest.raises(ValueError, match="no probe entry featurized successfully"):
        fp.feature_blocks(["invalid_smiles"])
