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
            {"type": "NullFeaturizer"},
            "NullFeaturizer",
            id="wrapper_params_omitted",
        ),
        pytest.param(NullFeaturizer(n_jobs=1), "NullFeaturizer", id="instance"),
    ],
)
def test_concatenator_list_entry_forms(entry, expected_name):
    """List forms must accept {type, params} wrappers and live featurizer instances."""
    # A concatenator needs 2+ featurizers, so pair the entry under test with a
    # filler of a class none of the parametrized entries use
    concat = FeatureConcatenator(
        featurizers=[
            entry,
            {"type": "DescriptorFeaturizer", "params": {"descr_type": "desc2d"}},
        ]
    )
    assert expected_name in [type(f).__name__ for f in concat.featurizers]


@pytest.mark.parametrize(
    "featurizers, duplicate",
    [
        pytest.param(
            [
                {"type": "FingerprintFeaturizer", "params": {"fp_type": "ecfp"}},
                {"type": "FingerprintFeaturizer", "params": {"fp_type": "ecfp:4"}},
            ],
            "FingerprintFeaturizer",
            id="same_class_unaliased",
        ),
        pytest.param(
            [
                {
                    "type": "FingerprintFeaturizer",
                    "alias": "fp",
                    "params": {"fp_type": "ecfp"},
                },
                {
                    "type": "FingerprintFeaturizer",
                    "alias": "fp",
                    "params": {"fp_type": "ecfp:4"},
                },
            ],
            "fp",
            id="repeated_alias",
        ),
        pytest.param(
            [
                {"type": "NullFeaturizer"},
                {
                    "type": "FingerprintFeaturizer",
                    "alias": "NullFeaturizer",
                    "params": {"fp_type": "ecfp"},
                },
            ],
            "NullFeaturizer",
            id="alias_shadows_other_type",
        ),
    ],
)
def test_concatenator_rejects_duplicate_block_keys(featurizers, duplicate):
    """Two blocks sharing a key are ambiguous for per-key transforms, however the collision arises."""
    with pytest.raises(ValidationError, match="same feature block key") as excinfo:
        FeatureConcatenator(featurizers=featurizers)

    # The message must name the offending key and the way out of the collision
    assert duplicate in str(excinfo.value)
    assert "alias" in str(excinfo.value)


def test_concatenator_rejects_duplicate_keys_across_nesting():
    """A nested child colliding with an outer sibling must fail at construction, not at transform fit."""
    with pytest.raises(ValidationError, match="same feature block key"):
        FeatureConcatenator(
            featurizers=[
                {
                    "type": "FingerprintFeaturizer",
                    "params": {"fp_type": "ecfp", "n_jobs": 1},
                },
                {
                    "type": "FeatureConcatenator",
                    "params": {
                        "featurizers": [
                            {
                                "type": "FingerprintFeaturizer",
                                "params": {"fp_type": "ecfp:6", "n_jobs": 1},
                            },
                            {"type": "NullFeaturizer"},
                        ]
                    },
                },
            ]
        )


def test_aliases_key_and_order_same_class_blocks(smiles):
    """Aliased same-class featurizers must combine, keying and ordering their blocks by alias."""
    concat = FeatureConcatenator(
        featurizers=[
            {
                "type": "FingerprintFeaturizer",
                "alias": "zz_maccs",
                "params": {"fp_type": "maccs", "n_jobs": 1},
            },
            {
                "type": "FingerprintFeaturizer",
                "alias": "aa_ecfp",
                "params": {"fp_type": "ecfp", "n_jobs": 1},
            },
        ]
    )
    feats, _ = concat.featurize(smiles)
    blocks = concat.feature_blocks()

    # Alias order, not class name, fixes the column layout
    assert [key for key, _ in blocks] == ["aa_ecfp", "zz_maccs"]
    assert concat.feature_block_keys() == ["aa_ecfp", "zz_maccs"]

    ecfp_feats, _ = concat.featurizers[0].featurize(smiles)
    maccs_feats, _ = concat.featurizers[1].featurize(smiles)
    assert concat.featurizers[0].fp_type == "ecfp"
    assert blocks == [
        ("aa_ecfp", ecfp_feats.shape[1]),
        ("zz_maccs", maccs_feats.shape[1]),
    ]
    assert (feats[:, : ecfp_feats.shape[1]] == ecfp_feats).all()
    assert (feats[:, ecfp_feats.shape[1] :] == maccs_feats).all()


def test_alias_accepted_inside_params():
    """An alias inside params must key the block the same as one written beside `type`."""
    concat = FeatureConcatenator(
        featurizers=[
            {"type": "NullFeaturizer", "params": {"alias": "first"}},
            {"type": "NullFeaturizer", "alias": "second"},
        ]
    )
    assert concat.feature_block_keys() == ["first", "second"]


def test_alias_given_twice_with_different_values_raises():
    """Two spellings of one entry's alias leave no honest choice and must be rejected."""
    with pytest.raises(ValidationError, match="give it once"):
        FeatureConcatenator(
            featurizers=[
                {
                    "type": "NullFeaturizer",
                    "alias": "beside",
                    "params": {"alias": "inside"},
                },
                {"type": "FingerprintFeaturizer", "params": {"fp_type": "ecfp"}},
            ]
        )


@pytest.mark.parametrize(
    "alias", [pytest.param("", id="empty"), pytest.param("   ", id="whitespace")]
)
def test_blank_alias_rejected(alias):
    """A blank alias would key a block to a name no transform config can reference."""
    with pytest.raises(ValidationError, match="non-empty name"):
        NullFeaturizer(alias=alias)


def test_alias_is_stripped():
    """Surrounding whitespace must not survive into the block key."""
    assert NullFeaturizer(alias="  padded  ").alias == "padded"


def test_concatenator_rejects_alias_on_itself():
    """A concatenator emits its children's blocks, so an alias on it would name nothing."""
    with pytest.raises(ValidationError, match="takes no alias"):
        FeatureConcatenator(
            alias="everything",
            featurizers=[
                {"type": "NullFeaturizer"},
                {
                    "type": "FingerprintFeaturizer",
                    "params": {"fp_type": "ecfp", "n_jobs": 1},
                },
            ],
        )


def test_concatenator_rejects_alias_on_nested_group():
    """Aliasing a nested group would reorder its children while naming none of their blocks."""
    with pytest.raises(ValidationError, match="takes no alias"):
        FeatureConcatenator(
            featurizers=[
                {"type": "NullFeaturizer"},
                {
                    "type": "FeatureConcatenator",
                    "alias": "fingerprints",
                    "params": {
                        "featurizers": [
                            {
                                "type": "FingerprintFeaturizer",
                                "params": {"fp_type": "ecfp", "n_jobs": 1},
                            },
                            {
                                "type": "DescriptorFeaturizer",
                                "params": {"descr_type": "desc2d", "n_jobs": 1},
                            },
                        ]
                    },
                },
            ]
        )


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"FingerprintFeaturizer": {"fp_type": "ecfp"}}, id="single_key"),
        pytest.param(
            {"FingerprintFeaturizer": {"fp_type": "ecfp"}, "extra": 1}, id="multi_key"
        ),
    ],
)
def test_concatenator_rejects_entry_without_type(entry):
    """Entries lacking `type` must be rejected, not silently resolved to a null type."""
    with pytest.raises(ValidationError, match="must be"):
        FeatureConcatenator(featurizers=[entry])


@pytest.mark.parametrize(
    "featurizers, match",
    [
        pytest.param(
            NullFeaturizer(n_jobs=1), "not a single featurizer", id="bare_instance"
        ),
        pytest.param([NullFeaturizer(n_jobs=1), 42], "got .*int", id="bad_entry_type"),
    ],
)
def test_concatenator_rejects_malformed_input_as_validation_error(featurizers, match):
    """Malformed input must surface as ValidationError, not an unwrapped TypeError.

    A bare featurizer is the case worth naming: pydantic models are iterable, so
    without the guard one would coerce to an empty list rather than fail.
    """
    with pytest.raises(ValidationError, match=match):
        FeatureConcatenator(featurizers=featurizers)


# The dict case only cares about the length check here; its deprecation warning
# is asserted by test_concatenator_dict_form_still_constructs
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "featurizers",
    [
        pytest.param([], id="empty_list"),
        pytest.param([{"type": "NullFeaturizer"}], id="single_entry"),
        pytest.param([NullFeaturizer(n_jobs=1)], id="single_instance"),
        pytest.param({"NullFeaturizer": {}}, id="single_dict_key"),
    ],
)
def test_concatenator_requires_two_featurizers(featurizers):
    """Concatenating fewer than two featurizers is a no-op and must be rejected, whatever the input shape."""
    with pytest.raises(ValidationError, match="at least 2 items"):
        FeatureConcatenator(featurizers=featurizers)


def test_concatenator_dict_form_still_constructs():
    """The deprecated dict form must keep working so saved recipe YAMLs stay loadable."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        concat = FeatureConcatenator(
            featurizers={
                "FingerprintFeaturizer": {"fp_type": "ecfp", "n_jobs": 1},
                "DescriptorFeaturizer": {"descr_type": "desc2d", "n_jobs": 1},
            }
        )
    names = [type(f).__name__ for f in concat.featurizers]
    assert names == ["DescriptorFeaturizer", "FingerprintFeaturizer"]


def test_concatenator_list_form_does_not_warn(recwarn):
    """The wrapper list form is the supported shape and must not emit a deprecation."""
    FeatureConcatenator(
        featurizers=[
            {"type": "NullFeaturizer", "params": {"n_jobs": 1}},
            {
                "type": "FingerprintFeaturizer",
                "params": {"fp_type": "ecfp", "n_jobs": 1},
            },
        ]
    )
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


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
    feats, idx = concat.featurize(smiles)
    blocks = concat.feature_blocks()

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
    feats, _ = outer.featurize(smiles)
    blocks = outer.feature_blocks()

    # Two featurizers, but three blocks: the nested concatenator contributes its
    # children rather than one block for itself
    assert len(outer.featurizers) == 2
    keys = [key for key, _ in blocks]
    assert keys == ["DescriptorFeaturizer", "FingerprintFeaturizer", "NullFeaturizer"]
    assert sum(width for _, width in blocks) == feats.shape[1]
    assert blocks[2] == ("NullFeaturizer", 1)

    # The static accessor must agree with what featurize actually emitted, since
    # the workflow checks per-block transform keys against it before featurizing
    assert outer.feature_block_keys() == keys


def test_concatenator_feature_blocks_before_featurize_raises():
    """feature_blocks must fail loudly rather than guess when featurize has not run yet."""
    concat = FeatureConcatenator(
        featurizers=[
            {
                "type": "FingerprintFeaturizer",
                "params": {"fp_type": "ecfp", "n_jobs": 1},
            },
            {"type": "NullFeaturizer", "params": {"n_jobs": 1}},
        ]
    )
    with pytest.raises(RuntimeError, match="featurize"):
        concat.feature_blocks()
