import numpy as np
import torch
from lightning import pytorch as pl
from pydantic import field_validator
from loguru import logger

from openadmet.models.architecture.model_base import TorchModelBase
from openadmet.models.architecture.model_base import models as model_registry

from typing import ClassVar, Optional, Any
from collections import OrderedDict


class NeuralPairwiseRegressor(pl.LightningModule):

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)

    def forward(self, x: torch.Tensor):
        return self.fnn(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], name: str):
        x_1, x_2, y = batch
        x = torch.cat((x_1, x_2), dim=1)
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log(f"{name}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "training")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "validation")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "testing")

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x_1, x_2, _ = batch
        x = torch.cat((x_1, x_2), dim=1)
        return self(x)

@model_registry.register("NepareModel")
class NepareModel(TorchModelBase):

    type: ClassVar[str] = "NepareModel"
    scaler: Optional[Any] = None

    # Model parameters
    input_dim: int = 128
    hidden_dim: int = 64
    num_layers: int = 3
    activation = torch.nn.ReLU
    n_targets: int = 1

    # Training parameters
    lr: float = 1e-3
    loss_function: str = "mse"

    @field_validator("loss_function")
    @classmethod
    def validate_loss_function(cls, value):
        """Validate loss function"""
        if value not in ["mse", "mae", "huber", "bce", "cross_entropy"]:
            raise ValueError("loss_function must be one of 'mse', 'mae', 'huber', 'bce', or 'cross_entropy'")
        return value

    @classmethod
    def from_params(cls, class_params: dict = None, model_params: dict = None):
        """
        Create model instance from parameters
        """

        if class_params:
            instance = cls(**class_params)
        else:
            instance = cls()

        instance.build()
        return instance

    def make_new(self) -> "NepareModel":
        """
        Create a new instance of the model with the same parameters
        """
        return self.__class__(**self.dict(exclude={"estimator"}))

    def train(self, dataloader, scaler=None):
        """
        Train the model
        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer"
        )

    def build(self, scaler=None):
        """
        Build the model
        """
        self.scaler = scaler

        if not self.estimator:

            model_config = {
                "input_size": self.input_dim * 2,
                "hidden_size": self.hidden_dim,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "lr": self.lr,
                "n_targets": self.n_targets
            }

            _modules = OrderedDict()
            for i in range(self.num_layers):
                _modules[f"hidden_{i}"] = torch.nn.Linear(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim)
                _modules[f"{self.activation.__name__.lower()}_{i}"] = self.activation()
            _modules["readout"] = torch.nn.Linear(self.hidden_dim, self.n_targets)
            self.estimator = torch.nn.Sequential(_modules)

            self._training_config = {
                "loss_function": self.loss_function,
                "optimizer": "AdamW",
                "lr": self.lr
            }

            logger.info(f"Built {self.type} with config: {model_config}")

        else:
            logger.warning(f"{self.type} already built, skipping build step")

    def predict(self, dataloader, accelerator="gpu", devices=1) -> np.ndarray:

        pass
