"""Shared helpers for anvil specification and workflow models."""

from __future__ import annotations

from typing import Any


def ensure_list(value: Any) -> Any:
    """
    Wrap a bare value into a one-element list.

    Sections that accept either a single entry or an ordered sequence share this
    normalization, so downstream code only handles the list form. None and
    existing lists pass through untouched, leaving pydantic to validate the
    element type.

    Parameters
    ----------
    value : Any
        The configured value, either a bare entry, a list, or None.

    Returns
    -------
    Any
        The value as a list, or None.

    """
    if value is None or isinstance(value, list):
        return value

    return [value]
