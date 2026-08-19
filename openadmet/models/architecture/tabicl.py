"""TabICL model implementations."""

from typing import ClassVar

import numpy as np
from loguru import logger
from pydantic import ConfigDict, Field

from openadmet.models.architecture.model_base import (
    PickleableModelBase,
    models,
    seed_to_sklearn_kwargs,
)

# TabICL's device kwarg accepts anything torch.device() parses (cpu, cuda,
# cuda:0, mps, xla, ...); these are the two spellings that differ from ours
_ACCELERATOR_ALIASES = {"gpu": "cuda", "tpu": "xla"}


def _resolve_device(accelerator: str) -> str | None:
    """
    Resolve an accelerator spelling to a value TabICL's ``device`` kwarg accepts.

    "auto" maps to ``None`` so TabICL runs its own cuda-availability check;
    trainer aliases map to torch device names, and every other value passes
    through verbatim.

    Parameters
    ----------
    accelerator : str
        The accelerator spelling to resolve.

    Returns
    -------
    str or None
        A value accepted by TabICL's ``device`` parameter.

    """
    if accelerator == "auto":
        return None
    return _ACCELERATOR_ALIASES.get(accelerator, accelerator)


class TabICLModelBase(PickleableModelBase):
    """
    Base class for TabICL models.

    Attributes
    ----------
    type : ClassVar[str]
        Model type identifier.
    accelerator : str
        Device to use for training and prediction. Mapped to ``device`` for TabICL.
    random_seed : int
        Random seed for reproducibility. Mapped to ``random_state`` for TabICL.
    n_estimators : int
        Number of ensemble members.
    batch_size : int
        Ensemble members processed together.
    use_amp : str
        Automatic mixed precision mode.
    use_fa3 : str
        Flash Attention 3 usage mode.
    offload_mode : str
        Offload mode for memory management.

    """

    model_config = ConfigDict(extra="allow")
    type: ClassVar[str]

    accelerator: str = Field(
        default="auto", description="The device to use for training and prediction."
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility.")
    n_estimators: int = Field(default=8, description="Number of ensemble members.")
    batch_size: int = Field(
        default=1, description="Ensemble batch size for memory control."
    )
    use_amp: str = Field(
        default="auto", description="Automatic mixed precision setting."
    )
    use_fa3: str = Field(default="auto", description="Flash Attention 3 setting.")
    offload_mode: str = Field(
        default="auto", description="Offload mode for large data."
    )

    @classmethod
    def _get_estimator_class(cls) -> type:
        """Return the TabICL estimator class."""
        raise NotImplementedError

    def _build_kwargs(self) -> dict:
        """Collect kwargs for the underlying estimator."""
        kwargs = seed_to_sklearn_kwargs(self.model_dump())
        kwargs["device"] = _resolve_device(kwargs.pop("accelerator"))
        return kwargs

    def build(self) -> None:
        """Prepare and build the model instance."""
        if not self.estimator:
            estimator_cls = self._get_estimator_class()
            kwargs = self._build_kwargs()
            self.estimator = estimator_cls(**kwargs)
        else:
            logger.warning("Model already exists, skipping build")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
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
        kwargs : dict
            Additional keyword arguments accepted for interface
            compatibility with the anvil pipeline (e.g. accelerator) and
            ignored; device placement is fixed at build time.

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


@models.register("TabICLRegressorModel")
class TabICLRegressorModel(TabICLModelBase):
    """TabICL regression model."""

    type: ClassVar[str] = "TabICLRegressorModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from tabicl import TabICLRegressor

        return TabICLRegressor


@models.register("TabICLClassifierModel")
class TabICLClassifierModel(TabICLModelBase):
    """TabICL classification model."""

    type: ClassVar[str] = "TabICLClassifierModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from tabicl import TabICLClassifier

        return TabICLClassifier

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

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
