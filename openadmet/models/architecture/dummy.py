from sklearn.dummy import DummyClassifier, DummyRegressor
from openadmet.models.architecture.model_base import PickleableModelBase, models
from typing import ClassVar
import numpy as np
from loguru import logger


class DummyModelBase(PickleableModelBase):


    type: ClassVar[str]
    mod_class: ClassVar[
        type
    ]  # To specify the XGBoost model class (e.g., XGBMRegressor or XGBMClassifier)
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
            Parameters for the XGBoost model class, such as n_estimators, max_depth,
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
            self.estimator = self.mod_class(**self.mod_params)
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





@models.register("DummyRegressorModel")
class DummyRegressorModel(DummyModelBase):
    """
    Dummy regression model

    Common parameters for dummy models can be found at:
    https://scikit-learn.org/stable/api/sklearn.dummy.html
    """
    type: ClassVar[str] = "DummyRegressorModel"
    mod_class: ClassVar[type] = DummyRegressor
    strategy: str = "mean"  # Default strategy for dummy models


    type: ClassVar[str] = "DummyRegressorModel"
    mod_class: ClassVar[type] = DummyRegressor



@models.register("DummyClassifierModel")
class DummyClassifierModel(DummyModelBase):
    """
    Dummy classification model

    Common parameters for dummy models can be found at:
    https://scikit-learn.org/stable/api/sklearn.dummy.html
    """
    type: ClassVar[str] = "DummyClassifierModel"
    mod_class: ClassVar[type] = DummyClassifier
    strategy: str = "most_frequent"  # Default strategy for dummy models
