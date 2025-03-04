from typing import Any
from lightning import pytorch as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from loguru import logger
from lightning.pytorch.loggers import WandbLogger
from openadmet_models.trainer.trainer_base import TrainerBase, trainers


@trainers.register("LightningTrainer")
class LightningTrainer(TrainerBase):
    """
    Trainer for sklearn models with grid search
    """

    max_epochs: int = 20
    devices: int = 1
    use_wandb: bool = False

    _logger: Any
    _trainer: Any


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare()


    def _prepare(self):
        """
        Build the model
        """
        # checkpointing = ModelCheckpoint(
        # "checkpoints",  # Directory where model checkpoints will be saved
        # "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        # "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        # mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        # save_last=True,  # Always save the most recent checkpoint, even if it's not the best
        # )
        if self.use_wandb:
            self._logger = WandbLogger(log_model="all")
        else:
            self._logger = False
        self._trainer = pl.Trainer(
        logger=self._logger,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=self.devices, # Use GPU if available
        max_epochs=self.max_epochs, # number of epochs to train for
        # callbacks=[checkpointing], # Use the configured checkpoint callback
        )

    

    def train(self, train_dataloader):
        """
        Train the model
        """
        logger.info(f"Training model {self.model._model}")
        self._trainer.fit(self.model._model, train_dataloader)
        return self.model

       