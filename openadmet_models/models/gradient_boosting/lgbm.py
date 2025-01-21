import lightgbm as lgb
import numpy as np
from typing import ClassVar
from openadmet_models.models.sklearn import PickleableModelBase
from openadmet_models.models.model_catalouge import register_model
import logging

logger = logging.getLogger(__name__)

@register_model
class LGBMRegressorModel(PickleableModelBase):
    """
    LightGBM regression model
    """
    type: ClassVar[str] = "LGBMRegressorModel"
    model_params: dict = {}

    @classmethod
    def from_params(cls, class_params: dict, model_params: dict):
        """
        Create a model from parameters
        """

        instance =  cls(**class_params, model_params=model_params)
        instance.build()
        return instance

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model
        """
        self.model.fit(X, y)


    def build(self):
        """
        Prepare the model
        """
        if not self.model:
            self.model = lgb.LGBMRegressor(**self.model_params)
        else:
            logger.info("Model already exists, skipping build")

        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        return self.model.predict(X)
    
