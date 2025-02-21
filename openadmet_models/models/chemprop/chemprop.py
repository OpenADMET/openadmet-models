from typing import ClassVar

import chemprop
import numpy as np
from loguru import logger

from openadmet_models.models.model_base import PickleableModelBase, models


@models.register("ChemPropSingleTaskModel")
class ChemPropSingleTaskModel(PickleableModelBase):
    """
    LightGBM regression model
    """

    type: ClassVar[str] = "ChemPropSingleTaskModel"
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
