from typing import ClassVar

import torch
from lightning import pytorch as pl
from loguru import logger
from mtenn.config import SchNetRepresentationConfig, ModelConfig

from openadmet.models.architecture.model_base import LightningModelBase
from openadmet.models.architecture.model_base import models as model_registry


# TODO: Inherit from LightningModuleBase to expose more configurability
class MTENNLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        loss_fn=torch.nn.MSELoss(),
        lr=1e-4,
        monitor_metric: str = "val_loss",
    ):
        super().__init__()
        self.model = model_config.build()
        self.loss_fn = loss_fn
        self.lr = lr
        self.monitor_metric = monitor_metric

    def forward(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        pred, _ = self.model(data)
        return pred

    def training_step(self, batch, batch_idx):
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
        data_batch, _ = batch
        preds = [self(data) for data in data_batch]
        return torch.cat(preds)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


@model_registry.register("MTENNSchNetModel")
class MTENNSchNetModel(LightningModelBase):
    """
    MTENN SchNet Model Implementation
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

    # Expose Model Config params (when adding other representations I will add param here; other params available)
    strategy: str = "concat"
    pred_readout: str = None
    weights_path: str = None

    def build(self, scaler=None):
        """
        Prepare the model
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
        pass

    def train(self, dataloader):
        """
        Train the model
        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer."
        )

    def predict(self, dataloader, accelerator="gpu", devices=1) -> torch.Tensor:
        """
        Use model for prediction
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
        Copy parameters to a new model instance without copying the estimator
        """
        return self.__class__(**self.dict(exclude={"estimator"}))
