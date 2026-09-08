"""Tests for the shared anvil helpers."""

import pytest

from openadmet.models.anvil.utils import ensure_list


def test_wraps_a_bare_value():
    """A single entry must become a one-element list."""
    assert ensure_list("PCATransform") == ["PCATransform"]


def test_passes_none_through():
    """None must stay None, so an unset optional section is not turned into [None]."""
    assert ensure_list(None) is None


def test_returns_an_existing_list_unchanged():
    """An already-sequenced value must keep its order and identity."""
    value = ["ImputeTransform", "PCATransform"]
    assert ensure_list(value) is value


def test_does_not_flatten_a_mapping():
    """A dict is one entry, not an iterable of keys."""
    assert ensure_list({"type": "PCATransform"}) == [{"type": "PCATransform"}]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param((1, 2), id="tuple"),
        pytest.param("", id="empty_string"),
        pytest.param(0, id="falsy_int"),
    ],
)
def test_wraps_anything_that_is_not_a_list_or_none(value):
    """Only list and None are passed through; every other value is wrapped."""
    assert ensure_list(value) == [value]
