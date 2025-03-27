from typing import ClassVar

import lightgbm as lgb
import numpy as np
from loguru import logger

from openadmet.models.architecture.model_base import PickleableModelBase, models


class LGBMModelBase(PickleableModelBase):
    """
    Base class for LightGBM models
    """

    type: ClassVar[str]
    model_class: ClassVar[
        type
    ]  # To specify the LightGBM model class (e.g., LGBMRegressor or LGBMClassifier)
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
            self.model = self.model_class(**self.model_params)
        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.model:
            raise ValueError("Model not trained")
        return self.model.predict(X)


@models.register("LGBMRegressorModel")
class LGBMRegressorModel(LGBMModelBase):
    """
    LightGBM regression model
    """

    type: ClassVar[str] = "LGBMRegressorModel"
    model_class: ClassVar[type] = lgb.LGBMRegressor


@models.register("LGBMClassifierModel")
class LGBMClassifierModel(LGBMModelBase):
    """
    LightGBM classification model
    """

    type: ClassVar[str] = "LGBMClassifierModel"
    model_class: ClassVar[type] = lgb.LGBMClassifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.model:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
