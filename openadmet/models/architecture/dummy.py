"""Dummy model implementations."""

from typing import ClassVar

import numpy as np
from loguru import logger
from sklearn.dummy import DummyClassifier, DummyRegressor

from openadmet.models.architecture.model_base import PickleableModelBase, models


class DummyModelBase(PickleableModelBase):
    """Base class for Dummy models, allows instantiation from parameters that are passable to the Dummy model classes."""

    # Meta parameters for this class
    type: ClassVar[str]
    mod_class: ClassVar[type]

    def build(self):
        """Prepare the model."""
        if not self.estimator:
            self.estimator = self.mod_class(**self.model_dump())
        else:
            logger.warning("Model already exists, skipping build")

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.

        Parameters
        ----------
        X: np.ndarray
            Training data features
        y: np.ndarray
            Training data labels

        """
        self.build()
        y_arr = np.asarray(y)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        self.estimator = self.estimator.fit(X, y_arr)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X: np.ndarray
            Featurized data to predict on
        kwargs: dict
            Additional keyword arguments to pass to the predict method of the Dummy model

        Returns
        -------
        np.ndarray
            Predictions from the model

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        pred = self.estimator.predict(X)
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=1)
        return pred


@models.register("DummyRegressorModel")
class DummyRegressorModel(DummyModelBase):
    """
    Dummy regression model.

    Common parameters for dummy models can be found at:
    https://scikit-learn.org/stable/api/sklearn.dummy.html
    """

    # Meta parameters for this class
    type: ClassVar[str] = "DummyRegressorModel"
    mod_class: ClassVar[type] = DummyRegressor

    # DummyRegressor parameters
    strategy: str = "mean"  # Default strategy for dummy models
    constant: float | None = None  # Default constant value for dummy models
    quantile: float | None = None  # Default quantile value for dummy models


@models.register("DummyClassifierModel")
class DummyClassifierModel(DummyModelBase):
    """
    Dummy classification model.

    Common parameters for dummy models can be found at:
    https://scikit-learn.org/stable/api/sklearn.dummy.html
    """

    # Meta parameters for this class
    type: ClassVar[str] = "DummyClassifierModel"
    mod_class: ClassVar[type] = DummyClassifier

    # DummyClassifier parameters
    strategy: str = "most_frequent"  # Default strategy for dummy models
    random_state: int | None = None  # Default random state for dummy models
    constant: int | str = None  # Default constant value for dummy models
