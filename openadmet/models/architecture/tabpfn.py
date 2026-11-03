"""TabPFN model implementations."""

import functools
from os import PathLike
from typing import ClassVar

import numpy as np
import torch
from loguru import logger
from pydantic import Field, field_validator

from openadmet.models.architecture.model_base import PickleableModelBase, models

# TabPFN's device kwarg accepts "auto" natively (DevicesSpecification = "auto");
# these are the two trainer aliases that differ from torch device names
_ACCELERATOR_ALIASES = {"gpu": "cuda", "tpu": "xla"}


def _resolve_device(accelerator: str) -> str:
    """
    Resolve an accelerator spelling to a value TabPFN's ``device`` kwarg accepts.

    ``"auto"`` passes through verbatim since TabPFN accepts it natively;
    trainer aliases map to torch device names, and every other value passes
    through verbatim.

    .. note::

        Unlike ``tabicl`` where ``"auto"`` maps to ``None`` (so TabICL runs its
        own device detection), TabPFN's ``DevicesSpecification`` resolves
        ``"auto"`` natively.  The two helpers share the same name and shape but
        have different semantics — take care when porting between them.

    Parameters
    ----------
    accelerator : str
        The accelerator spelling to resolve.

    Returns
    -------
    str
        A value accepted by TabPFN's ``device`` parameter.

    """
    return _ACCELERATOR_ALIASES.get(accelerator, accelerator)


class TabPFNModelBase(PickleableModelBase):
    """
    Base class for TabPFN models using the basic TabPFN implementation.

    Attributes
    ----------
    accelerator : str
        Device to use for training and prediction. Mapped to ``device`` for TabPFN.
    random_seed : int
        Random seed for reproducibility. The legacy ``random_state`` name is
        accepted as a deprecated alias.
    ignore_pretraining_limits : bool
        Whether to ignore pretraining limits of TabPFN base models.

    """

    # Meta parameters for this class
    type: ClassVar[str]

    @classmethod
    def _get_estimator_class(cls) -> type:
        """Return the basic TabPFN estimator class (deferred import)."""
        raise NotImplementedError

    # TabPFN parameters
    accelerator: str = Field(default="auto")
    random_seed: int = Field(default=42)
    ignore_pretraining_limits: bool = Field(default=False)

    @field_validator("accelerator")
    @classmethod
    def validate_accelerator(cls, value: str) -> str:
        """Reject accelerator spellings ``torch.device()`` cannot parse."""
        resolved = _resolve_device(value)
        if resolved != "auto":
            try:
                torch.device(resolved)
            except RuntimeError as e:
                raise ValueError(f"Invalid accelerator {value!r}: {e}") from e
        return value

    def build(self):
        """Prepare and build the model instance."""
        accelerator = _resolve_device(self.accelerator)
        if not self.estimator:
            self.estimator = self._get_estimator_class()(
                device=accelerator,
                random_state=self.random_seed,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
            )
        else:
            logger.warning("Model already exists, skipping build")

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training targets.

        """
        self.build()
        self.estimator = self.estimator.fit(X, y)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        kwargs: Dict
            Keyword arguments for prediciton.

        Returns
        -------
        np.ndarray
            Model predictions with shape (n_samples, 1).

        Raises
        ------
        ValueError
            If the estimator does not exist.

        """
        if not self.estimator:
            raise ValueError("The estimator is not instantiated. Please call `build()` or `train()` first.")
        return np.expand_dims(self.estimator.predict(X), axis=1)

    def load(self, path: PathLike):
        """
        Load the model from a json, mapping any torch state to CPU.

        Parameters
        ----------
        path: PathLike
            Path to load the model from

        """
        self.estimator = self._get_estimator_class().load_model(path)

    def save(self, path: PathLike):
        """
        Save the model to a file.

        Parameters
        ----------
        path: PathLike
            Path to save the model to

        """
        self.estimator = self._get_estimator_class().save_model(path)


@models.register("TabPFNRegressorModel")
class TabPFNRegressorModel(TabPFNModelBase):
    """TabPFN regression model using the basic `tabpfn` implementation."""

    # Meta parameters for this class
    type: ClassVar[str] = "TabPFNRegressorModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor


@models.register("TabPFNClassifierModel")
class TabPFNClassifierModel(TabPFNModelBase):
    """TabPFN classification model using the basic `tabpfn` implementation."""

    # Meta parameters for this class
    type: ClassVar[str] = "TabPFNClassifierModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier
