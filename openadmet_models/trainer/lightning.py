from typing import Any
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from openadmet_models.trainer.trainer_base import TrainerBase, trainers
from pathlib import Path

@trainers.register("LightningTrainer")
class LightningTrainer(TrainerBase):
    """
    Trainer for sklearn models with grid search
    """

    max_epochs: int = 20
    accelerator: str = "gpu"
    devices: int = -1
    use_wandb: bool = False
    output_dir: Path = None
    wandb_project: str = "openadmet-testing"

    wandb_logger: Any = None
    _logger: Any
    _trainer: Any




    def prepare(self):
        """
        Build the model trainer
        """
    
        checkpointing = ModelCheckpoint(
        self.output_dir / "checkpoints",  # Directory where model checkpoints will be saved
        "best-{epoch}-{train_loss:.4f}",  # Filename format for checkpoints, including epoch and validation loss
        "train_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        save_top_k=1,  # Keep the top 1 checkpoints
        )
        self._logger = []
        if self.use_wandb:
            self.wandb_logger = WandbLogger(log_model=True, save_dir=self.output_dir, project=self.wandb_project)
            self._logger.append(self.wandb_logger)
        self._logger.append(CSVLogger(self.output_dir / "logs", name="model"))
        self._trainer = pl.Trainer(
        logger=self._logger,
        enable_progress_bar=True,
        accelerator=self.accelerator,
        devices=self.devices, # Use GPU if available
        max_epochs=self.max_epochs, # number of epochs to train for
        callbacks=[checkpointing], # Use the configured checkpoint callback
        )



    

    def train(self, train_dataloader):
        """
        Train the model
        """
        logger.info(f"Training model {self.model._model}")
        self._trainer.fit(self.model._model, train_dataloader)
        return self.model

       