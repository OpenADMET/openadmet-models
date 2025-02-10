from sklearn import pipeline
from sklearn.utils.discovery import all_estimators
from typing import ClassVar

import numpy as np
from loguru import logger

from openadmet_models.models.model_base import PickleableModelBase, models


def get_sklearn_estimators_as_dict(type_filter: str = None):
    """
    Get the sklearn estimators

    Parameters
    ----------
    type_filter: str
        Filter for the type of estimator to get, one of “classifier”, “regressor”, “cluster”, “transformer”
    """
    estimators = all_estimators(type_filter=type_filter)
    estimator_dict = {name: est for name, est in estimators}
    return estimator_dict



@models.register("SKLearnPipelineModel")
class SKLearnPipelineModel(PickleableModelBase):
    """
    SKLearn pipeline Regression model
    """

    type: ClassVar[str] = "SKLearnPipeline"
    model_params: dict = {}

    @classmethod
    def from_params(cls, class_params: dict = {}, model_params: dict = {}):
        """
        Create a model from parameters
        """

        instance = cls(**class_params, model_params=model_params)
        instance.build()
        return instance

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model
        """
        self.build()
        self.model = self.model.fit(X, y)

    def build(self):
        """
        Prepare the model
        """
        if not self.model:
            self.model = 
        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.model:
            raise ValueError("Model not trained")
        return self.model.predict(X)
