from typing import ClassVar

import numpy as np
import torch
from chemprop import models, nn
from lightning import pytorch as pl
from loguru import logger

from openadmet_models.models.model_base import PickleableModelBase
from openadmet_models.models.model_base import models as model_registry

_METRIC_TO_LOSS = {"mae": nn.metrics.MAE(), "rmse": nn.metrics.RMSE()}


@model_registry.register("ChemPropSingleTaskRegressorModel")
class ChemPropSingleTaskRegressorModel(PickleableModelBase):
    """
    ChemProp regression model
    """

    type: ClassVar[str] = "ChemPropSingleTaskModel"
    batch_norm: bool = True
    metric_list: list = ["mae"]
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
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer"
        )

    def build(self, scaler=None):
        """
        Prepare the model
        """
        if not self.model:
            if scaler is not None:
                output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
            else:
                output_transform = None

            metric_list = [_METRIC_TO_LOSS[metric] for metric in self.metric_list]
            mpnn = models.MPNN(
                nn.BondMessagePassing(),
                nn.MeanAggregation(),
                nn.RegressionFFN(output_transform=output_transform),
                self.batch_norm,
                metric_list,
            )
            self._model = mpnn

        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.model:
            raise AttributeError("Model not trained")

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None, enable_progress_bar=False, accelerator="auto", devices=1
            )
            preds = trainer.predict(self.model, X)
        # concatenate the predictions which are in a list of tensors
        return torch.cat(preds).numpy().ravel()
