from typing import ClassVar

import lightgbm as lgb
import numpy as np
from loguru import logger

from openadmet.models.architecture.model_base import PickleableModelBase, models


@models.register("LGBMRegressorModel")
class LGBMRegressorModel(PickleableModelBase):
    """
    LightGBM regression model
    """

    type: ClassVar[str] = "LGBMRegressorModel"
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
        if not self.estimator:
            self.estimator = lgb.LGBMRegressor(**self.model_params)
        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict(X)
