from typing import ClassVar

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from loguru import logger

from openadmet.models.architecture.model_base import PickleableModelBase, models


class RFModelBase(PickleableModelBase):
    """
    Base class for Sklearn Random Forest models, allows instantiation from parameters that are passable to the RF model classes.
    """

    type: ClassVar[str]
    mod_class: ClassVar[
        type
    ]  # To specify the  model class (e.g., RandomForestRegressor or RandomForestClassifier)
    mod_params: dict = {}

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters

        Parameters
        ----------
        class_params: dict
            Parameters for the model class, such as type, mod_class, etc.
        mod_params: dict
            Parameters for the Random Forest model class, such as n_estimators, max_depth,
            learning_rate, etc.

        """
        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model

        Parameters
        ----------
        X: np.ndarray
            Training data features
        y: np.ndarray
            Training data labels

        """
        self.build()
        self.estimator = self.estimator.fit(X, y, verbose=True)

    def build(self):
        """
        Prepare the model
        """
        if not self.estimator:
            self.estimator = self.mod_class(**self.mod_params, n_jobs=-1)
        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model

        Parameters
        ----------
        X: np.ndarray
            Data to predict on

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
    """
    Random Forest regression model
    """

    type: ClassVar[str] = "RFRegressorModel"
    mod_class: ClassVar[type] = RandomForestRegressor


@models.register("RFClassifierModel")
class RFClassifierModel(RFModelBase):
    """ """

    type: ClassVar[str] = "RFClassifierModel"
    mod_class: ClassVar[type] = RandomForestClassifier

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
