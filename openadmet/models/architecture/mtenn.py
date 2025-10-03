"""MTENN model implementation."""

from typing import ClassVar

import torch
from lightning import pytorch as pl
from loguru import logger
from mtenn.config import ModelConfig, SchNetRepresentationConfig

from openadmet.models.architecture.model_base import LightningModelBase
from openadmet.models.architecture.model_base import models as model_registry


class MTENNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for MTENN models.

    Parameters
    ----------
    model_config : ModelConfig
        MTENN ModelConfig instance containing representation, strategy,and readout configuration.
    loss_fn : torch.nn.Module, default=torch.nn.MSELoss()
        Loss function used for training.
    lr : float, default=1e-4
        Learning rate
    monitor_metric : str, default="val_loss"
        The metric to monitor during training/validation.

    """

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
        Prediction step for Lightning Trainer.

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
        Configure AdamW optimizer for training. This will eventually run through calling LightningModuleBase.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer for training.

        """
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


@model_registry.register("MTENNSchNetModel")
class MTENNSchNetModel(LightningModelBase):
    """
    MTENN SchNet Model Implementation.

    Class to implement a MTENN based model, specifically one using the SchNet Representation.
    This exposes the hyperparameters and model-level options directly to the anvil workflow.
    Future versions of this class will enable other types of representations.

    Parameters
    ----------
    hidden_channels : int, default=128
        Hidden embedding size of SchNet.
    num_filters : int, default=128
        Number of filters in cfconv layers.
    num_interactions : int, default=6
        Number of interaction blocks.
    num_gaussians : int, default=50
        Number of Gaussians for distance expansion.
    cutoff : float, default=10.0
        Cutoff distance for interactions.
    max_num_neighbors : int, default=32
        Maximum neighbors considered per atom.
    readout : str, default="add"
        Global aggregation method ("add" or "mean").

    strategy : str, default="concat"
        MTENN strategy for combining representations ("delta", "concat", "complex").
    pred_readout : str or None, default=None
        Readout function for predictions ("pic50", "pki", or None).
    weights_path : str or None, default=None
        Optional path to load pretrained weights.

    """

    type: ClassVar[str] = "MTENNSchNetModel"

    # Expose Schnet Representation hyper params
    hidden_channels: int = 128
    num_filters: int = 128
    num_interactions: int = 6
    num_gaussians: int = 50
    cutoff: float = 10.0
    max_num_neighbors: int = 32
    readout: str = "add"

    # Expose Model Config params
    strategy: str = "concat"
    pred_readout: str = None
    weights_path: str = None

    def build(self, scaler=None):
        """
        Prepare and build the model.

        Parameters
        ----------
        scaler : object, optional
            Scaler for data normalization (default: None).

        """
        if not self.estimator:
            model_rep = SchNetRepresentationConfig(
                hidden_channels=self.hidden_channels,
                num_filters=self.num_filters,
                num_interactions=self.num_interactions,
                num_gaussians=self.num_gaussians,
                cutoff=self.cutoff,
                max_num_neighbors=self.max_num_neighbors,
                readout=self.readout,
            )
            model_config = ModelConfig(
                representation=model_rep,
                strategy=self.strategy,
                pred_readout=self.pred_readout,
                weights_path=self.weights_path,
            )
            self.estimator = MTENNLightningModule(model_config)
        else:
            logger.warning("Model already exists, skipping build.")

    # Deprecated now; remove from anvil workflow eventually?
    def from_params(self, params):
        """Set for compatability with anvil workflow. Method deprecated."""
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
            Training is not implemented in the model class; use LightningTrainer.

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
            Concatenated predictions from all batches.

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
