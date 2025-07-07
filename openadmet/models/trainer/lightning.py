from pathlib import Path #it is used in the main therefore i do not remove it
from typing import Any

from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger
from pydantic import model_validator

from openadmet.models.trainer.trainer_base import TrainerBase, trainers


@trainers.register("LightningTrainer")
class LightningTrainer(TrainerBase):
    """
    Trainer for sklearn models with grid search
    """

    max_epochs: int = 20
    accelerator: str = "gpu"
    devices: int = 1
    use_wandb: bool = False
    output_dir: Path = None
    wandb_project: str = "openadmet-testing"
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_mode: str = "min"
    early_stopping_min_delta: float = 0.001
    monitor_metric: str = "val_loss"

    wandb_logger: Any = None
    _logger: Any
    _trainer: Any
    _callbacks: Any = None

    @model_validator(mode="after")
    def check_monitor_metric(self):
        """
        Check if the monitor metric is valid
        """
        if self.monitor_metric not in ["val_loss", "train_loss"]:
            raise ValueError(
                f"Invalid monitor metric: {self.monitor_metric}"
                ". Must be one of ['val_loss', 'train_loss']"
            )
        return self

    def prepare(self):
        """
        Build the model trainer
        """

        # Initialize logging container
        self._logger = []

        # initialize the callbacks list
        self._callbacks = []

        fmtstring = (
            "best-{epoch}-{val_loss:.4f}"
            if self.monitor_metric == "val_loss"
            else "best-{epoch}-{train_loss:.4f}"
        )

        # Configure checkpoint callbacks
        checkpointing = ModelCheckpoint(
            self.output_dir
            / "checkpoints",  # Directory where model checkpoints will be saved
            fmtstring,  # Filename format for checkpoints, including epoch and validation loss
            self.monitor_metric,  # Metric used to select the best checkpoint (based on validation loss)
            mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
            save_last=True,  # Always save the most recent checkpoint, even if it's not the best
            save_top_k=1,  # Keep the top 1 checkpoints
        )

        if self.early_stopping:
            # Import EarlyStopping callback if early stopping is enabled

            # Configure early stopping callback
            early_stopping_callback = EarlyStopping(
                min_delta=self.early_stopping_min_delta,  # Minimum change in the monitored quantity to qualify as an improvement
                monitor=self.monitor_metric,  # Monitor validation loss for early stopping
                patience=self.early_stopping_patience,  # Number of epochs with no improvement after which training will be stopped
                mode=self.early_stopping_mode,  # Stop when validation loss stops decreasing
            )
            self._callbacks.append(early_stopping_callback)

        # Append the checkpointing callback to the callbacks list
        self._callbacks.append(checkpointing)

        # Append wandb longer if requested
        if self.use_wandb:
            self.wandb_logger = WandbLogger(
                log_model=True, save_dir=self.output_dir, project=self.wandb_project
            )
            self._logger.append(self.wandb_logger)

        # Append CSV logger
        self._logger.append(CSVLogger(self.output_dir / "logs", name="model"))

        # Initialize the PyTorch Lightning trainer
        self._trainer = pl.Trainer(
            logger=self._logger,
            enable_progress_bar=True,
            accelerator=self.accelerator,
            devices=self.devices,  # Use GPU if available
            max_epochs=self.max_epochs,  # number of epochs to train for
            callbacks=[checkpointing],  # Use the configured checkpoint callback
        )

    def train(self, train_dataloader, val_dataloader):
        """
        Train the model
        """

        # Indicate that the model is being trained
        logger.debug(f"Training model {self.model._estimator}")

        # Fit model
        self._trainer.fit(self.model._estimator, train_dataloader, val_dataloader)

        return self.model

import torch
import torch.nn as nn
from torch_geometric.data import Batch
@trainers.register("GATv2LightningTrainer")
class GATv2LightningTrainer(LightningTrainer):
    """
    Trainer for GATv2 models using PyTorch Lightning.
    """

    # Training hyperparameters, now part of the trainer
    loss_function: str = "mse"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10

    def train(self, train_dataloader, val_dataloader):
        """
        Train the GATv2 model.
        """
        from openadmet.models.architecture.gat import GATv2Model

        # The model passed in should be a GATv2ModelWrapper instance
        if not isinstance(self.model.estimator, GATv2Model):
            raise TypeError(f"GATv2LightningTrainer can only train GATv2Model, but got {type(self.model.estimator)}")

        # Prepare model config from the model wrapper
        model_config = {
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "num_heads": self.model.num_heads,
            "dropout": self.model.gat_dropout,
            "pooling": self.model.pooling,
            "output_dim": self.model.output_dim,
            "edge_dim": self.model.edge_dim,
            "concat_heads": self.model.concat_heads,
            "add_self_loops": self.model.add_self_loops,
            "share_weights": self.model.share_weights,
            "bias": self.model.bias,
        }

        # Prepare training config from this trainer's own attributes
        training_config = {
            "loss_fn_name": self.loss_function,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "scheduler_name": self.scheduler,
            "warmup_epochs": self.warmup_epochs,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_patience": self.scheduler_patience,
        }

        # Create the Lightning wrapper
        lightning_wrapper = GATv2LightningWrapper(
            model_config=model_config,
            **training_config
        )

        logger.debug(f"Training GATv2 model with Lightning wrapper: {lightning_wrapper}")

        # Fit the model
        self._trainer.fit(lightning_wrapper, train_dataloader, val_dataloader)

        # Copy trained weights back to the core model
        self.model.estimator.load_state_dict(lightning_wrapper.model.state_dict())

        return self.model


class GATv2LightningWrapper(pl.LightningModule):
    """
    Lightning wrapper for GAT model
    """
    def __init__(
        self,
        model_config: dict,
        loss_fn_name: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_name: str = "cosine",
        warmup_epochs: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn_name'])

        self.loss_functions = {
            "mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss(),
            "bce": nn.BCEWithLogitsLoss(), "cross_entropy": nn.CrossEntropyLoss()
        }

        # Import GATv2Model here to avoid circular imports
        from openadmet.models.architecture.gat import GATv2Model
        self.model = GATv2Model(**model_config)
        self.loss_fn = self._get_loss_function(loss_fn_name)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.warmup_epochs = warmup_epochs
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

    def _get_loss_function(self, name: str):
        if name.lower() not in self.loss_functions:
            raise ValueError(f"Unsupported loss function: {name}. Supported: {list(self.loss_functions.keys())}")
        return self.loss_functions[name.lower()]

    def forward(self, data: Batch):
        """Forward pass"""
        return self.model(data)

    def training_step(self, batch: Batch, batch_idx: int):
        """Training step"""
        target = batch.y

        pred = self(batch)

        if pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)

        loss = self.loss_fn(pred, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        """Validation step"""
        target = batch.y

        pred = self(batch)

        if pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)

        loss = self.loss_fn(pred, target)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def predict_step(self, batch: Batch, batch_idx: int):
        """Prediction step"""
        data = batch
        pred = self(data)
        return pred

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        elif self.scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.scheduler_name == "none":
            return optimizer
        else:
            logger.warning(f"Unsupported scheduler: {self.scheduler_name}, using AdamW without LR scheduler.")
            return optimizer
