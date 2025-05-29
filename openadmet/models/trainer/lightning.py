from pathlib import Path
from typing import Any

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
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


        fmtstring = "best-{epoch}-{val_loss:.4f}" if self.monitor_metric == "val_loss" else "best-{epoch}-{train_loss:.4f}"

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
                monitor=self.monitor_metric,  # Monitor validation loss for early stopping
                patience=self.early_stopping_patience,  # Number of epochs with no improvement after which training will be stopped
                mode=self.early_stopping_mode,  # Stop when validation loss stops decreasing
            )
            self._logger.append(early_stopping_callback)

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
