"""Tests for consolidated random-seed handling (random_seed + random_state alias)."""

import warnings

import pytest

from openadmet.models.anvil.specification import (
    ModelSpec,
    SplitSpec,
    _resolve_model_init_seed,
    _section_sets_seed,
)
from openadmet.models.architecture.model_base import seed_to_sklearn_kwargs
from openadmet.models.split.sklearn import ShuffleSplitter


def test_random_seed_set_directly_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        splitter = ShuffleSplitter(random_seed=7)
    assert splitter.random_seed == 7


def test_section_maps_deprecated_random_state_with_warning():
    with pytest.warns(DeprecationWarning, match="random_state"):
        spec = SplitSpec(type="ShuffleSplitter", params={"random_state": 7})
    assert spec.params["random_seed"] == 7
    assert "random_state" not in spec.params


def test_section_explicit_random_seed_wins_over_alias():
    with pytest.warns(DeprecationWarning):
        spec = SplitSpec(
            type="ShuffleSplitter", params={"random_seed": 1, "random_state": 2}
        )
    assert spec.params["random_seed"] == 1


def test_seed_to_sklearn_kwargs_renames_key():
    assert seed_to_sklearn_kwargs({"random_seed": 5, "n_estimators": 10}) == {
        "random_state": 5,
        "n_estimators": 10,
    }


def test_seed_to_sklearn_kwargs_no_seed_is_untouched():
    # A regressor (e.g. SVR) exposes no seed; the helper must not inject random_state
    assert seed_to_sklearn_kwargs({"C": 1.0}) == {"C": 1.0}


def test_section_sets_seed_detects_random_seed():
    assert _section_sets_seed(
        SplitSpec(type="ShuffleSplitter", params={"random_seed": 1})
    )
    assert not _section_sets_seed(SplitSpec(type="ShuffleSplitter", params={}))


def test_resolve_model_init_seed_precedence():
    explicit = ModelSpec(type="RFRegressorModel", params={"random_seed": 11})
    assert _resolve_model_init_seed(explicit, global_seed=99) == 11

    no_section = ModelSpec(type="RFRegressorModel", params={})
    assert _resolve_model_init_seed(no_section, global_seed=99) == 99
