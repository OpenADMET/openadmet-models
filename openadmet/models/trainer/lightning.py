"""PyTorch Lightning trainer implementation."""

from pathlib import Path  # it is used in the main therefore i do not remove it
from typing import Any

from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger

from openadmet.models.trainer.trainer_base import TrainerBase, trainers


@trainers.register("LightningTrainer")
class LightningTrainer(TrainerBase):
    """
    Trainer for sklearn models with grid search.

    Attributes
    ----------
    max_epochs : int
        The maximum number of epochs to train for.
    accelerator : str
        The accelerator to use, e.g. 'cpu', 'gpu'.
    devices : int
        The number of devices to use, e.g. 1 for single GPU, -1
        for all available GPUs.
    use_wandb : bool
        Whether to use Weights & Biases for logging.
    output_dir : Path
        The output directory to save logs and models.
    wandb_project : str
        The Weights & Biases project name.
    early_stopping : bool
        Whether to use early stopping.
    early_stopping_patience : int
        The number of epochs with no improvement after which
        training will be stopped.
    early_stopping_mode : str
        The mode for early stopping, either 'min' or 'max'.
    early_stopping_min_delta : float
        The minimum change in the monitored quantity to qualify
        as an improvement.
    gradient_clip_val : float
        The value to clip gradients at.
    precision : int
        The precision to use, e.g. 32, 16, or 'bf16'.
    accumulate_grad_batches : int
        The number of batches to accumulate gradients over.
    deterministic : bool
        Whether to use deterministic algorithms.
    fast_dev_run : bool
        Whether to run a single batch for debugging.
    limit_train_batches : float
        The fraction of training batches to use, e.g. 1.0 for all,
        0.5 for half, or an integer for a fixed number.
    limit_val_batches : float
        The fraction of validation batches to use, e.g. 1.0 for all,
        0.5 for half, or an integer for a fixed number. Default is 1.0.
    wandb_logger : Any
        The Weights & Biases logger.
    _logger : Any
        The logger.
    _trainer : Any
        The PyTorch Lightning trainer.
    _callbacks : Any
        The callbacks.

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
    gradient_clip_val: float = 0.0
    precision: int = 32
    accumulate_grad_batches: int = 1
    deterministic: bool = False
    fast_dev_run: bool = False
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0

    wandb_logger: Any = None
    _logger: Any
    _trainer: Any
    _callbacks: Any = None

    def build(self):
        """Build the model trainer."""
        # Initialize logging container
        self._logger = []

        # initialize the callbacks list
        self._callbacks = []

        fmtstring = (
            "best-{epoch}-{val_loss:.4f}"
            if self.model.estimator.monitor_metric == "val_loss"
            else "best-{epoch}-{train_loss:.4f}"
        )

        # Configure checkpoint callbacks
        checkpointing = ModelCheckpoint(
            self.output_dir
            / "checkpoints",  # Directory where model checkpoints will be saved
            fmtstring,  # Filename format for checkpoints, including epoch and validation loss
            self.model.estimator.monitor_metric,  # Metric used to select the best checkpoint (based on validation loss)
            mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
            save_last=True,  # Always save the most recent checkpoint, even if it's not the best
            save_top_k=1,  # Keep the top 1 checkpoints
        )

        # Append the checkpointing callback to the callbacks list
        self._callbacks.append(checkpointing)

        # Configure early stopping callback
        if self.early_stopping:
            early_stopping_callback = EarlyStopping(
                min_delta=self.early_stopping_min_delta,  # Minimum change in the monitored quantity to qualify as an improvement
                monitor=self.model.estimator.monitor_metric,  # Monitor validation loss for early stopping
                patience=self.early_stopping_patience,  # Number of epochs with no improvement after which training will be stopped
                mode=self.early_stopping_mode,  # Stop when validation loss stops decreasing
            )
            self._callbacks.append(early_stopping_callback)

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
            callbacks=self._callbacks,
            gradient_clip_val=self.gradient_clip_val,
            precision=self.precision,
            accumulate_grad_batches=self.accumulate_grad_batches,
            deterministic=self.deterministic,
            fast_dev_run=self.fast_dev_run,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
        )

    def train(self, train_dataloader, val_dataloader):
        """
        Train the model.

        Parameters
        ----------
        train_dataloader : DataLoader
            The training data loader.
        val_dataloader : DataLoader
            The validation data loader.

        Returns
        -------
        model : TrainerBase
            The trained model.

        """
        # Indicate that the model is being trained
        logger.debug(f"Training model {self.model.estimator}")

        # Fit model
        self._trainer.fit(self.model.estimator, train_dataloader, val_dataloader)

        return self.model

    def make_new(self) -> "LightningTrainer":
        """Copy parameters to a new LightningTrainer instance."""
        return self.__class__(**self.__dict__)
