from typing import ClassVar

import numpy as np
import torch
from chemprop import models, nn
from torch.nn import Identity
from lightning import pytorch as pl
from loguru import logger

from pydantic import field_validator

from openadmet.models.architecture.model_base import TorchModelBase
from openadmet.models.architecture.model_base import models as model_registry

_METRIC_TO_LOSS = {"mse": nn.metrics.MSE(), "mae": nn.metrics.MAE(), "rmse": nn.metrics.RMSE()}


@model_registry.register("ChemPropSingleTaskRegressorModel")
class ChemPropSingleTaskRegressorModel(TorchModelBase):
    """
    ChemProp regression model
    """

    type: ClassVar[str] = "ChemPropSingleTaskModel"
    batch_norm: bool = True
    metric_list: list = ["mse", "mae", "rmse"]
    mod_params: dict = {}
    depth: int = 3
    message_hidden_dim: int = 300
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 1
    messages: str = "bond"
    aggregation: str = "norm"
    normalized_targets: bool = True


    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value):
        """
        Validate the messages parameter
        """
        if value not in ["bond", "atom"]:
            raise ValueError("Messages must be either 'bond' or 'atom'")
        return value


    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, value):
        """
        Validate the aggregation parameter
        """
        if value not in ["mean", "norm"]:
            raise ValueError("Aggregation must be either 'mean' or 'norm'")
        return value


    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters
        """

        instance = cls(**class_params, mod_params=mod_params)
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
        if not self.estimator:
            if scaler is not None:
                output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
            elif self.normalized_targets:
                # expects the targets to be normalized, likely to be loaded from state dict
                output_transform = nn.UnscaleTransform([1], [0])
            else:
                output_transform = Identity()

            metric_list = [_METRIC_TO_LOSS[metric] for metric in self.metric_list]


            aggregation_cls = nn.MeanAggregation if self.aggregation == "mean" else nn.NormAggregation
            message_cls = nn.BondMessagePassing if self.messages == "bond" else nn.AtomMessagePassing

            # Create the model
            mp = message_cls(d_h=self.message_hidden_dim, depth=self.depth)
            aggr = aggregation_cls()

            ffn = nn.RegressionFFN(input_dim=self.message_hidden_dim, hidden_dim=self.ffn_hidden_dim, n_layers=self.ffn_num_layers, output_transform=output_transform)
            # Create the MPNN model

            mpnn = models.MPNN(mp, aggr, ffn, self.batch_norm, metric_list)

            self.estimator = mpnn

        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray, accelerator="gpu", devices=1, **kwargs) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.estimator:
            raise AttributeError("Model not trained")

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None, enable_progress_bar=False, accelerator=accelerator, devices=devices
            )
            preds = trainer.predict(self.estimator, X)
        return torch.cat(preds).numpy()
