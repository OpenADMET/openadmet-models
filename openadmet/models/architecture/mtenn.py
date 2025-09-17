"""MTENN model implementation."""

from typing import ClassVar

import torch
from lightning import pytorch as pl
from loguru import logger
from mtenn.config import SchNetRepresentationConfig, ModelConfig

from openadmet.models.architecture.model_base import LightningModelBase
from openadmet.models.architecture.model_base import models as model_registry


class MTENNLightningModule(pl.LightningModule):
    """MTENN Lightning Module."""

    def __init__(
        self,
        model_config: ModelConfig,
        loss_fn=torch.nn.MSELoss(),
        lr=1e-4,
        monitor_metric: str = "val_loss",
    ):
        """
        Initialize the MTENN Lightning Module.

        Parameters
        ----------
        model_config : ModelConfig
            Configuration for the MTENN model.
        loss_fn : callable, optional
            Loss function to use (default: torch.nn.MSELoss()).
        lr : float, optional
            Learning rate (default: 1e-4).
        monitor_metric : str, optional
            Metric to monitor during training, can be "val_loss" or "train_loss" (default: "val_loss").

        """
        super().__init__()
        self.model = model_config.build()
        self.loss_fn = loss_fn
        self.lr = lr
        self.monitor_metric = monitor_metric

    def forward(self, data):
        """
        Forward pass.

        Parameters
        ----------
        data : dict
            Input data as a dictionary of tensors.

        Returns
        -------
        torch.Tensor
            Model predictions.

        """
        for k, v in data.items():
            data[k] = v.to(self.device)
        pred, _ = self.model(data)
        return pred

    def training_step(self, batch, batch_idx):
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
        data_batch, target_batch = batch
        batch_loss = 0.0

        for data, target in zip(data_batch, target_batch):
            pred = self(data)
            loss = self.loss_fn(pred, target.unsqueeze(0).to(self.device))
            batch_loss += loss

        avg_loss = batch_loss / len(data_batch)
        self.log("train_loss", avg_loss)
        return avg_loss

    def predict_step(self, batch, batch_idx):
        """
        Perform a prediction step.

        Parameters
        ----------
        batch : tuple
            Tuple containing a batch of input data and targets.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Concatenated predictions for the batch.

        """
        data_batch, _ = batch
        preds = [self(data) for data in data_batch]
        return torch.cat(preds)

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer for training.

        """
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


@model_registry.register("MTENNSchNetModel")
class MTENNSchNetModel(LightningModelBase):
    """MTENN SchNet Model Implementation."""

    type: ClassVar[str] = "MTENNSchNetModel"
    mod_params: dict = {}

    def build(self, scaler=None):
        """
        Prepare and build the model.

        Parameters
        ----------
        scaler : object, optional
            Scaler for data normalization (default: None).

        """
        if not self.estimator:
            model_rep = SchNetRepresentationConfig(**self.mod_params)
            model_config = ModelConfig(
                representation=model_rep, strategy="delta", pred_readout="pic50"
            )
            self.estimator = MTENNLightningModule(model_config)
        else:
            logger.warning("Model already exists, skipping build.")

    def from_params(self, params):
        """
        Load model parameters from a dictionary.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters.

        """
        pass

    def train(self, dataloader):
        """
        Train the model.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for training data.

        Raises
        ------
        NotImplementedError
            This method is not implemented; use a trainer instead.

        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer."
        )

    def predict(self, dataloader, accelerator="gpu", devices=1) -> torch.Tensor:
        """
        Use the model for prediction.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for prediction data.
        accelerator : str, optional
            Accelerator type, e.g., "gpu" or "cpu" (default: "gpu").
        devices : int, optional
            Number of devices to use (default: 1).

        Returns
        -------
        np.ndarray
            Model predictions as a NumPy array.

        Raises
        ------
        AttributeError
            If the model is not built or trained.

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

    def make_new(self) -> "MTENNSchNetModel":
        """
        Copy parameters to a new model instance without copying the estimator.

        Returns
        -------
        MTENNSchNetModel
            A new instance of MTENNSchNetModel with the same parameters.

        """
        return self.__class__(**self.mod_params, **self.dict(exclude={"estimator"}))
