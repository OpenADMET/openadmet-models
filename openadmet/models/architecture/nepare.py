import numpy as np
import torch
from lightning import pytorch as pl
from pydantic import field_validator
from loguru import logger
from chemprop import models, nn

from openadmet.models.architecture.model_base import models as model_registry
from openadmet.models.architecture.model_base import LightningModelBase

from nepare.nn import NeuralPairwiseRegressor

from typing import ClassVar

@model_registry.register("NeuralPairwiseRegressorModel")
class NeuralPairwiseRegressorModel(LightningModelBase):
    """
    NepareChemPropModel is a neural pairwise regression model based on ChemProp.
    It uses learned embeddings for pairwise features.
    """

    type: ClassVar[str] = "NeuralPairwiseRegressorModel"
    mod_params: dict = {}
    monitor_metric: str = "val_loss"

    def train(self, dataloader):
        """
        Train the model
        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer."
        )
    
    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters
        """

        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

    def build(self, scaler=None):
        if not self.estimator:
            nepare = NeuralPairwiseRegressor(**self.mod_params)
            self.estimator = nepare
        else:
            logger.warning("Model already exists, skipping build")

        return self

    def make_new(self) -> "NeuralPairwiseRegressorModel":
        """
        Copy parameters to a new model instance without copying the estimator
        """
        return self.__class__(**self.mod_params, **self.dict(exclude={"estimator"}))

    def predict(self, dataloader, accelerator="gpu", devices=1) -> torch.Tensor:

        if not self.estimator:
            raise AttributeError("Model not built or trained.")
        
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=False,
                accelerator=accelerator,
                devices=devices,
            )
            preds = trainer.predict(self.estimator, dataloader)
        return torch.cat(preds, dim=0).numpy()
