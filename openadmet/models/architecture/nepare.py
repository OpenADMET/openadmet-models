"""Neural Pairwise Regressor Model implementation."""

import numpy as np
import torch
from lightning import pytorch as pl
from pydantic import field_validator
from loguru import logger
from chemprop import models, nn
from collections import OrderedDict

from openadmet.models.architecture.model_base import models as model_registry
from openadmet.models.architecture.model_base import (
    LightningModuleBase,
    LightningModelBase,
)


from typing import ClassVar, Any, Optional


class NeuralPairwiseRegressorModule(LightningModuleBase):
    """Neural Pairwise Regressor Module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        lr: float = 1e-4,
        n_targets: int = 1,
        monitor_metric: str = "val_loss",
        scaler=None,
    ):
        """
        Initialize the Neural Pairwise Regressor Module.

        Attributes
        ----------
        input_dim : int
            Size of the input features for a single molecule.
        hidden_dim : int
            Size of the hidden layers.
        num_layers : int
            Number of hidden layers.
        activation : str, optional
            Activation function to use, options are relu or gelu.
        lr : float, optional
            Learning rate (default: 1e-4).
        n_targets : int, optional
            Number of target outputs (default: 1).
        monitor_metric : str, optional
            Metric to monitor during training, can be "val_loss" or "train_loss" (
            default: "val_loss").
        scaler : object, optional
            Scaler for data normalization (default: None).

        """
        super().__init__()
        input_dim = input_dim * 2
        if activation == "relu":
            activation = torch.nn.ReLU
        elif activation == "gelu":
            activation = torch.nn.GELU
        _modules = OrderedDict()
        for i in range(num_layers):
            _modules[f"hidden_{i}"] = torch.nn.Linear(
                input_dim if i == 0 else hidden_dim, hidden_dim
            )
            _modules[f"{activation.__name__.lower()}_{i}"] = activation()
        _modules["readout"] = torch.nn.Linear(hidden_dim, n_targets)
        self.fnn = torch.nn.Sequential(_modules)
        self.lr = lr
        self.save_hyperparameters()

    def __hash__(self):
        """Hash based on object id to ensure uniqueness in sets and dicts."""
        return id(self)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim * 2).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_targets).

        """
        return self.fnn(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        Perform a training step.

        Parameters
        ----------
        batch : tuple
            Tuple containing a batch of input data and targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            The average loss for the batch.

        """
        return self._step(batch, "train_loss")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        Perform a validation step.

        Parameters
        ----------
        batch : tuple
            Tuple containing a batch of input data and targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            The average loss for the batch.

        """
        return self._step(batch, "val_loss")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        Perform a test step.

        Parameters
        ----------
        batch : tuple
            Tuple containing a batch of input data and targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            The average loss for the batch.

        """
        return self._step(batch, self.monitor_metric)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], name: str):
        """
        Perform a training/validation/test step.

        Parameters
        ----------
        batch : tuple
            Tuple containing a batch of input data and targets.
        name : str
            Name of the metric to log.

        Returns
        -------
        torch.Tensor
            The average loss for the batch.

        """
        x_1, x_2, y = batch
        x = torch.cat((x_1, x_2), dim=1)
        y_hat = self(x)
        if y.dim() == 1:
            y = y.unsqueeze(1)  # Ensure y is [batch_size, 1]
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log(f"{name}", loss, prog_bar=True)
        return loss

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Perform a prediction step.

        Parameters
        ----------
        batch : tuple
            Tuple containing a batch of input data and targets.

        Returns
        -------
        torch.Tensor
            Concatenated predictions for the batch.

        """
        if len(batch) == 3:
            x_1, x_2, _ = batch
        else:
            x_1, x_2 = batch
        x = torch.cat((x_1, x_2), dim=1)
        y_hat = self(x)
        return y_hat


@model_registry.register("NeuralPairwiseRegressorModel")
class NeuralPairwiseRegressorModel(LightningModelBase):
    """
    NepareChemPropModel is a neural pairwise regression model based on ChemProp.

    It uses learned embeddings for pairwise features.
    """

    type: ClassVar[str] = "NeuralPairwiseRegressorModel"
    input_dim: int = 1028
    hidden_dim: int = 128
    num_layers: int = 3
    activation: str = "relu"
    lr: float = 1e-4
    n_targets: int = 1
    monitor_metric: str = "val_loss"
    scaler: Optional[Any] = None

    def train(self, dataloader):
        """
        Train the model.

        Parameters
        ----------
        dataloader : DataLoader
            The training data loader.

        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer."
        )

    def build(self, scaler=None, input_dim=None):
        """
        Prepare and build the model.

        Parameters
        ----------
        scaler : object, optional
            Scaler for data normalization (default: None).
        input_dim : int, optional
            Size of the input features for a single molecule (default: None).

        Returns
        -------
        self : NeuralPairwiseRegressorModel
            The built model instance.

        """
        self.scaler = scaler if scaler is not None else self.scaler
        self.input_dim = input_dim if input_dim is not None else self.input_dim

        if not self.estimator:
            self.estimator = NeuralPairwiseRegressorModule(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                activation=self.activation,
                lr=self.lr,
                n_targets=self.n_targets,
                monitor_metric=self.monitor_metric,
                scaler=self.scaler,
            )
        else:
            logger.warning("Model already exists, skipping build")

        return self

    def predict(self, dataloader, accelerator="gpu", devices=1) -> torch.Tensor:
        """
        Predict using the model.

        Parameters
        ----------
        dataloader : DataLoader
            The data loader for prediction.
        accelerator : str, optional
            Accelerator type (default: "gpu").
        devices : int, optional
            Number of devices to use (default: 1).

        Returns
        -------
        np.ndarray
            Predictions as a NumPy array.

        """
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
