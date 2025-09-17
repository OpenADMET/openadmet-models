"""TabPFN model implementations."""

from typing import ClassVar, Literal, Optional, Union

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNRegressor,
    AutoTabPFNClassifier,
)
from tabpfn import TabPFNRegressor, TabPFNClassifier
import numpy as np
from loguru import logger
from pydantic import field_validator, Field

from openadmet.models.architecture.model_base import PickleableModelBase, models


class TabPFNExtensionModelBase(PickleableModelBase):
    """
    Base class for TabPFN models using the tabpfn-extensions package.

    This class provides common functionality for TabPFN models with post-hoc ensembling,
    including configuration, building, training, and prediction.

    Attributes
    ----------
    type : ClassVar[str]
        Model type identifier.
    mod_class : ClassVar[type]
        The TabPFN model class (e.g., AutoTabPFNRegressor or AutoTabPFNClassifier).
    max_time : Optional[int]
        Maximum time to spend on fitting the post hoc ensemble.
    accelerator : Literal["cpu", "gpu", "auto"]
        Device to use for training and prediction.
    random_state : int
        Random seed for reproducibility.
    ignore_pretraining_limits : bool
        Whether to ignore pretraining limits of TabPFN base models.
    phe_init_args : Optional[dict]
        Initialization arguments for the post hoc ensemble predictor.

    """

    type: ClassVar[str]
    mod_class: ClassVar[type]

    max_time: Optional[int] = Field(
        default=None,
        description="The maximum time to spend on fitting the post hoc ensemble.",
    )

    accelerator: Literal["cpu", "gpu", "auto"] = Field(
        default="auto", description="The device to use for training and prediction."
    )

    random_state: int = Field(
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
    def validate_accelerator(cls, value):
        """
        Validate the accelerator parameter.

        Parameters
        ----------
        value : str
            The accelerator value to validate.

        Returns
        -------
        str
            The validated accelerator value.

        Raises
        ------
        ValueError
            If the accelerator is not one of 'cpu', 'gpu', or 'auto'.

        """
        if value not in ["cpu", "gpu", "auto"]:
            raise ValueError("Accelerator must be either 'cpu' or 'gpu' or 'auto'")

        return value

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters.

        Parameters
        ----------
        class_params : dict, optional
            Class-level parameters.
        mod_params : dict, optional
            Model-specific parameters.

        Returns
        -------
        TabPFNExtensionModelBase
            An instance of the model.

        """
        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

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

    def build(self):
        """Prepare and build the model instance."""
        accelerator = self.accelerator if self.accelerator != "gpu" else "cuda"
        if not self.estimator:
            self.estimator = self.mod_class(
                max_time=self.max_time,
                device=accelerator,
                random_state=self.random_state,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
                phe_init_args=self.phe_init_args,
            )
        else:
            logger.warning("Model already exists, skipping build")

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

        Raises
        ------
        ValueError
            If the model is not trained.

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("TabPFNPostHocRegressorModel")
class TabPFNPostHocRegressorModel(TabPFNExtensionModelBase):
    """TabPFN regression model using `tabpfn-extensions` with posthoc ensembling."""

    type: ClassVar[str] = "TabPFNPostHocRegressorModel"
    mod_class: ClassVar[type] = AutoTabPFNRegressor


@models.register("TabPFNPostHocClassifierModel")
class TabPFNPostHocClassifierModel(TabPFNExtensionModelBase):
    """TabPFN classification model using `tabpfn-extensions` with posthoc ensembling."""

    type: ClassVar[str] = "TabPFNPostHocClassifierModel"
    mod_class: ClassVar[type] = AutoTabPFNClassifier

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

        Raises
        ------
        ValueError
            If the model is not trained.

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)


class TabPFNModelBase(PickleableModelBase):
    """
    Base class for TabPFN models using the basic tabpfn implementation.

    Attributes
    ----------
    accelerator : Literal["cpu", "cuda", "auto"]
        Device to use for training and prediction.
    random_state : int
        Random seed for reproducibility.
    ignore_pretraining_limits : bool
        Whether to ignore pretraining limits of TabPFN base models.

    """

    accelerator: Literal["cpu", "cuda", "auto"] = Field(default="auto")
    random_state: int = Field(default=42)
    ignore_pretraining_limits: bool = Field(default=False)

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters.

        Parameters
        ----------
        class_params : dict, optional
            Class-level parameters.
        mod_params : dict, optional
            Model-specific parameters.

        Returns
        -------
        TabPFNModelBase
            An instance of the model.

        """
        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

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

    def build(self):
        """Prepare and build the model instance."""
        accelerator = self.accelerator if self.accelerator != "gpu" else "cuda"
        if not self.estimator:
            self.estimator = self.mod_class(
                device=accelerator,
                random_state=self.random_state,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
            )
        else:
            logger.warning("Model already exists, skipping build")

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

    type: ClassVar[str] = "TabPFNRegressorModel"
    mod_class: ClassVar[type] = TabPFNRegressor


@models.register("TabPFNClassifierModel")
class TabPFNClassifierModel(TabPFNModelBase):
    """TabPFN classification model using the basic `tabpfn` implementation."""

    type: ClassVar[str] = "TabPFNClassifierModel"
    mod_class: ClassVar[type] = TabPFNClassifier
