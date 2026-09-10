"""TabPFN model implementations."""

import warnings
from typing import ClassVar, Optional

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


class TabPFNExtensionModelBase(PickleableModelBase):
    """
    Base class for TabPFN models using the tabpfn-extensions package.

    This class provides common functionality for TabPFN models with post-hoc ensembling,
    including configuration, building, training, and prediction.

    Attributes
    ----------
    type : ClassVar[str]
        Model type identifier.
    max_time : Optional[int]
        Maximum time to spend on fitting the post hoc ensemble.
    accelerator : str
        Device to use for training and prediction. Mapped to ``device`` for TabPFN.
    random_seed : int
        Random seed for reproducibility. The legacy ``random_state`` name is
        accepted as a deprecated alias.
    ignore_pretraining_limits : bool
        Whether to ignore pretraining limits of TabPFN base models.
    phe_init_args : Optional[dict]
        Initialization arguments for the post hoc ensemble predictor.

    """

    # Meta parameters for this class
    type: ClassVar[str]

    @classmethod
    def _get_estimator_class(cls) -> type:
        """Return the TabPFN extension estimator class (deferred import)."""
        raise NotImplementedError

    # TabPFN parameters
    max_time: Optional[int] = Field(
        default=None,
        description="The maximum time to spend on fitting the post hoc ensemble.",
    )
    accelerator: str = Field(
        default="auto", description="The device to use for training and prediction."
    )
    random_seed: int = Field(
        default=42,
        description="Controls both the randomness of base models and the post hoc ensembling method.",
    )
    ignore_pretraining_limits: bool = Field(
        default=False,
        description="Whether to ignore the pretraining limits of the TabPFN base models.",
    )
    phe_init_args: Optional[dict] = Field(
        default=None,
        description="The initialization arguments for the post hoc ensemble predictor. "
        "See post_hoc_ensembles.pfn_phe.AutoPostHocEnsemblePredictor for more options and all details.",
    )

    @field_validator("accelerator")
    @classmethod
    def validate_accelerator(cls, value: str) -> str:
        """
        Validate the accelerator parameter.

        TabPFN's ``device`` kwarg uses ``DevicesSpecification`` which accepts
        any torch device name or ``"auto"``.  This validator reuses the same
        resolution path so a bad accelerator is caught eagerly at construction
        time rather than many calls later at ``fit()``.
        """
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
        warnings.warn(
            "TabPFN 2.5 is distributed under the TabPFN 2.5 License: https://priorlabs.ai/tabpfn-license which prohibits commercial use. Review the license and ensure you are compliant before using this model. A commercial license can be obtained from the TabPFN team."
        )
        if not self.estimator:
            self.estimator = self._get_estimator_class()(
                max_time=self.max_time,
                device=accelerator,
                random_state=self.random_seed,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
                phe_init_args=self.phe_init_args,
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
        Predict on data using the model.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        kwargs: Dict
            Keyword arguments for model

        Returns
        -------
        np.ndarray
            Model predictions with shape (n_samples, 1).

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("TabPFNPostHocRegressorModel")
class TabPFNPostHocRegressorModel(TabPFNExtensionModelBase):
    """TabPFN regression model using `tabpfn-extensions` with posthoc ensembling."""

    # Meta parameters for this class
    type: ClassVar[str] = "TabPFNPostHocRegressorModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
            AutoTabPFNRegressor,
        )

        return AutoTabPFNRegressor


@models.register("TabPFNPostHocClassifierModel")
class TabPFNPostHocClassifierModel(TabPFNExtensionModelBase):
    """TabPFN classification model using `tabpfn-extensions` with posthoc ensembling."""

    # Meta parameters for this class
    type: ClassVar[str] = "TabPFNPostHocClassifierModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
            AutoTabPFNClassifier,
        )

        return AutoTabPFNClassifier

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the model.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)


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
        """Reject accelerator spellings ``torch.device()`` cannot parse.

        Mirrors ``TabPFNExtensionModelBase.validate_accelerator`` so both the
        basic and extension model families fail eagerly on a bad accelerator.
        """
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
            If the model is not trained.

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


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
