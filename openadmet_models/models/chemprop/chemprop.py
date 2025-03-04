from typing import ClassVar

import chemprop
import numpy as np
from loguru import logger
from chemprop import  models
from chemprop import nn
from openadmet_models.models.model_base import PickleableModelBase
from openadmet_models.models.model_base import models as model_registry

@model_registry.register("ChemPropSingleTaskRegressorModel")
class ChemPropSingleTaskRegressorModel(PickleableModelBase):
    """
    LightGBM regression model
    """

    type: ClassVar[str] = "ChemPropSingleTaskModel"
    batch_norm: bool = True
    metric_list: list = [nn.metrics.MAE(), nn.metrics.RMSE()]
    model_params: dict = {}

    @classmethod
    def from_params(cls, class_params: dict = {}, model_params: dict = {}):
        """
        Create a model from parameters
        """

        instance = cls(**class_params, model_params=model_params)
        instance.build()
        return instance

    def train(self, dataloader, scaler=None):
        """
        Train the model
        """
        raise NotImplementedError("Training not implemented in model class, use a trainer")

    def build(self, scaler=None):
        """
        Prepare the model
        """
        if not self.model:
            if scaler is not None:
                output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
            else:
                output_transform = None
            mpnn = models.MPNN(nn.BondMessagePassing(), nn.MeanAggregation(), nn.RegressionFFN(output_transform=output_transform), self.batch_norm, self.metric_list)
            self._model = mpnn

        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.model:
            raise ValueError("Model not trained")
        return self.model.predict(X)
