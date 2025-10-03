"""Random Forest model implementations."""

from typing import ClassVar

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from openadmet.models.architecture.model_base import PickleableModelBase, models


class RFModelBase(PickleableModelBase):
    """Base class for Sklearn Random Forest models."""

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
        self.estimator = self.estimator.fit(X, y, verbose=True)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X: np.ndarray
            Data to predict on
        **kwargs
            Additional keyword arguments for the predict method.

        Returns
        -------
        np.ndarray
            Predictions from the model

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("RFRegressorModel")
class RFRegressorModel(RFModelBase):
    """Random Forest regression model."""

    # Meta parameters for this class
    type: ClassVar[str] = "RFRegressorModel"
    mod_class: ClassVar[type] = RandomForestRegressor

    # RF parameters
    n_estimators: int = 100
    criterion: str = "squared_error"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: float = 1.0
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int | None = None
    random_state: int | None = None
    verbose: int = 0
    warm_start: bool = False
    ccp_alpha: float = 0.0
    max_samples: float | None = None
    monotonic_cst: float | None = None


@models.register("RFClassifierModel")
class RFClassifierModel(RFModelBase):
    """RF classifier model."""

    # Meta parameters for this class
    type: ClassVar[str] = "RFClassifierModel"
    mod_class: ClassVar[type] = RandomForestClassifier

    # RF parameters
    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: float | str = "sqrt"
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int | None = None
    random_state: int | None = None
    verbose: int = 0
    warm_start: bool = False
    class_weight: dict | None = None
    ccp_alpha: float = 0.0
    max_samples: float | None = None
    monotonic_cst: float | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model, returning probabilities for each class.

        Parameters
        ----------
        X: np.ndarray
            Data to predict on

        Returns
        -------
        np.ndarray
            Probabilities for each class from the model

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)
