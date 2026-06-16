"""
Shared random-seed handling for registry components.

Components expose a single ``random_seed`` field. The legacy ``random_state``
name is still accepted as input for backwards compatibility, mapped onto
``random_seed`` with a :class:`DeprecationWarning`. Internal serialization always
emits ``random_seed``, so reloading a component never triggers the warning.
"""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import BaseModel, model_validator

DEFAULT_RANDOM_SEED: int = 42


class RandomSeedMixin(BaseModel):
    """
    Mixin that accepts the deprecated ``random_state`` alias for ``random_seed``.

    Subclasses declare their own ``random_seed`` field (with the default that
    suits the component); this mixin only rewrites incoming ``random_state``
    keys and warns. Apply it solely to classes that declare ``random_seed``.
    """

    @model_validator(mode="before")
    @classmethod
    def _map_deprecated_random_state(cls, data: Any) -> Any:
        """Map a deprecated ``random_state`` key onto ``random_seed``."""
        if not isinstance(data, dict) or "random_state" not in data:
            return data

        data = dict(data)
        legacy = data.pop("random_state")
        warnings.warn(
            "`random_state` is deprecated; use `random_seed` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # An explicit random_seed takes precedence over the legacy alias
        data.setdefault("random_seed", legacy)
        return data


def seed_to_sklearn_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """
    Rewrite a ``random_seed`` key to scikit-learn's ``random_state``.

    Models that forward ``model_dump()`` straight into a scikit-learn estimator
    expose ``random_seed`` to users but must hand the estimator its native
    ``random_state`` argument.

    Parameters
    ----------
    params : dict
        Estimator keyword arguments, typically from ``model_dump()``.

    Returns
    -------
    dict
        The same mapping with ``random_seed`` renamed to ``random_state``.

    """
    params = dict(params)
    if "random_seed" in params:
        params["random_state"] = params.pop("random_seed")
    return params
