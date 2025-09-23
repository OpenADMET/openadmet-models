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


from typing import ClassVar


_METRIC_TO_LOSS = {
    "mse": nn.metrics.MSE(),
    "mae": nn.metrics.MAE(),
    "rmse": nn.metrics.RMSE(),
}


class NeuralPairwiseRegressorModule(LightningModuleBase):
    """Neural Pairwise Regressor Module."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        activation=torch.nn.ReLU,
        lr: float = 1e-4,
        n_targets: int = 1,
        monitor_metric: str = "val_loss",
    ):
        """
        Initialize the Neural Pairwise Regressor Module.

        Attributes
        ----------
        input_size : int
            Size of the input features for a single molecule.
        hidden_size : int
            Size of the hidden layers.
        num_layers : int
            Number of hidden layers.
        activation : callable, optional
            Activation function to use (default: torch.nn.ReLU).
        lr : float, optional
            Learning rate (default: 1e-4).
        n_targets : int, optional
            Number of target outputs (default: 1).
        monitor_metric : str, optional
            Metric to monitor during training, can be "val_loss" or "train_loss" (
            default: "val_loss").

        """
        super().__init__()
        input_size = input_size * 2
        _modules = OrderedDict()
        for i in range(num_layers):
            _modules[f"hidden_{i}"] = torch.nn.Linear(
                input_size if i == 0 else hidden_size, hidden_size
            )
            _modules[f"{activation.__name__.lower()}_{i}"] = activation()
        _modules["readout"] = torch.nn.Linear(hidden_size, n_targets)
        self.fnn = torch.nn.Sequential(_modules)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size * 2).

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
        return self._step(batch, self.monitor_metric)

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
        return self._step(batch, self.monitor_metric)

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
        x_1, x_2, _ = batch
        x = torch.cat((x_1, x_2), dim=1)
        return self(x)


@model_registry.register("NeuralPairwiseRegressorModel")
class NeuralPairwiseRegressorModel(LightningModelBase):
    """
    NepareChemPropModel is a neural pairwise regression model based on ChemProp.

    It uses learned embeddings for pairwise features.
    """

    type: ClassVar[str] = "NeuralPairwiseRegressorModel"
    mod_params: dict = {}

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

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters.

        Parameters
        ----------
        class_params : dict, optional
            Parameters for the model class (default: {}).
        mod_params : dict, optional
            Parameters for the model module (default: {}).

        Returns
        -------
        NeuralPairwiseRegressorModel
            An instance of the model.

        """
        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

    def build(self, scaler=None):
        """
        Prepare and build the model.

        Parameters
        ----------
        scaler : object, optional
            Scaler for data normalization (default: None).

        Returns
        -------
        self : NeuralPairwiseRegressorModel
            The built model instance.

        """
        if not self.estimator:
            nepare = NeuralPairwiseRegressorModule(**self.mod_params)
            self.estimator = nepare
        else:
            logger.warning("Model already exists, skipping build")

        return self

    def make_new(self) -> "NeuralPairwiseRegressorModel":
        """Copy parameters to a new model instance without copying the estimator."""
        return self.__class__(**self.mod_params, **self.dict(exclude={"estimator"}))

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
