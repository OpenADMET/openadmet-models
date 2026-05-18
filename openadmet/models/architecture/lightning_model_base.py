"""Lightning-specific base classes for deep learning models.

This module is intentionally separate from model_base so that importing
PickleableModelBase (sklearn-style models) does not incur the cost of
loading torch or lightning.pytorch.
"""

import json
from abc import abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Any, ClassVar

import torch
from lightning import pytorch as pl
from pydantic import field_validator

from openadmet.models.architecture.model_base import ModelBase
from openadmet.models.drivers import DriverType


@dataclass
class LightningModuleBase(pl.LightningModule):
    """
    Lightning module base class.

    A PyTorch lightning model may inherit this instead of pl.LightningModule
    to preconfigure optimizer and scheduler.
    """

    # Meta parameters for this class
    type: ClassVar[str]

    # Optimizer and scheduler configuration
    optimizer: str = "adamw"
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-5
    scheduler: str = "cosine"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    monitor_metric: str = "val_loss"

    def __post_init__(self):
        """Defer initialization of the LightningModuleBase."""
        pl.LightningModule.__init__(self)

    @field_validator("monitor_metric")
    @classmethod
    def check_monitor_metric(cls, value):
        """Check if the monitor metric is valid."""
        allowed = ["val_loss", "train_loss"]
        if value.lower() not in allowed:
            raise ValueError(f"Monitored metric must be one of {allowed}")
        return value

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, value):
        """Validate the optimizer parameter."""
        allowed = {"adamw", "adam", "sgd"}
        if value.lower() not in allowed:
            raise ValueError(f"Optimizer must be one of {allowed}")
        return value

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, value):
        """Validate the scheduler parameter."""
        allowed = {"cosine", "reduce_on_plateau", "none", None}
        if (value.lower() not in allowed) and (value is not None):
            raise ValueError(f"Scheduler must be one of {allowed}")
        return value

    def configure_optimizers(self):
        """Return optimizer and scheduler configuration for Lightning's configure_optimizers."""
        # Adamw optimizer
        if self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizer_lr,
                weight_decay=self.optimizer_weight_decay,
            )

        # Adam optimizer
        elif self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_lr,
                weight_decay=self.optimizer_weight_decay,
            )

        # SGD optimizer
        elif self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.optimizer_lr,
                weight_decay=self.optimizer_weight_decay,
            )

        # Cosine scheduler
        if self.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=10,  # T_max could be exposed as a parameter
            )

            scheduler_config = {
                "scheduler": scheduler,
                "monitor": self.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            }

        # Reduce on plateau scheduler
        elif self.scheduler.lower() == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )

            scheduler_config = {
                "scheduler": scheduler,
                "monitor": self.monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            }

        # No scheduler
        elif (self.scheduler is None) or (self.scheduler.lower() == "none"):
            scheduler_config = None

        # Return optimizer and scheduler configuration
        if scheduler_config:
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        else:
            return optimizer


class LightningModelBase(ModelBase):
    """A model that uses PyTorch Lightning."""

    # Meta parameters for this class
    type: ClassVar[str]
    _model_save_name: ClassVar[str] = "model.pth"
    _driver_type: DriverType = DriverType.LIGHTNING

    def make_new(self):
        """
        Copy parameters to a new model instance without copying the estimator.

        Returns
        -------
        LightningModelBase
            A new instance of LightningModelBase with the same parameters.

        """
        return self.__class__(**self.model_dump(exclude={"estimator"}))

    def save(self, path: PathLike):
        """
        Save the model to a file.

        Parameters
        ----------
        path: PathLike
            Path to save the model to

        """
        torch.save(self.estimator.state_dict(), path)

    def load(self, path: PathLike):
        """
        Load the model from a file.

        Parameters
        ----------
        path: PathLike
            Path to load the model from

        """
        self.estimator.load_state_dict(torch.load(path, weights_only=True))

    def serialize(
        self, param_path: PathLike = "model.json", serial_path: PathLike = "model.pth"
    ):
        """
        Save the model to a json file and a serialized file.

        Parameters
        ----------
        param_path: PathLike
            Path to save the model parameters to
        serial_path: PathLike
            Path to save the serialized model to

        """
        with open(param_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        self.save(serial_path)

    @classmethod
    def deserialize(
        cls,
        param_path: PathLike = "model.json",
        serial_path: PathLike = "model.pth",
        scaler: Any = None,
    ):
        """
        Create a model from parameters and a serialized model.

        Parameters
        ----------
        param_path: PathLike
            Path to load the model parameters from
        serial_path: PathLike
            Path to load the serialized model from
        scaler: Any, optional
            Scaler for target normalization, if applicable

        Returns
        -------
        instance: LightningModelBase
            An instance of the LightningModelBase class

        """
        with open(param_path) as f:
            mod_params = json.load(f)
        instance = cls(**mod_params)
        instance.build(scaler=scaler)
        instance.load(serial_path)
        return instance

    def freeze_weights(self, *args, **kwargs):
        """
        Freeze parts of the model for transfer learning or fine-tuning.

        Parameters
        ----------
        *args: variable length argument list
            Arguments to be passed to the implementing model's `freeze_weights` method.
        **kwargs: keyword arguments
            Keyword arguments to be passed to the implementing model's `freeze_weights` method.

        Notes
        -----
        This method should set the `requires_grad` attribute of the specified layers to False,
        preventing their weights from being updated during training. It also should set these
        layers to evaluation mode.

        """
        raise NotImplementedError(f"Weight freezing not implemented for {self.type}.")
