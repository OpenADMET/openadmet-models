import json
from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, ClassVar

import joblib
import torch
from class_registry import ClassRegistry, RegistryKeyError
from lightning import pytorch as pl
from pydantic import BaseModel, field_validator

models = ClassRegistry(unique=True)


def get_mod_class(model_type):
    try:
        feat_class = models.get_class(model_type)
    except RegistryKeyError:
        raise ValueError(f"Model type {model_type} not found in model catalouge")
    return feat_class


class ModelBase(BaseModel, ABC):
    _estimator: Any = None

    _model_json_name: ClassVar[str] = "model.json"

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @abstractmethod
    def from_params(cls, class_params: dict, mod_params: dict):
        """
        Create a model from parameters, abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def build(self):
        """
        Prepare the model, abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def save(self, path: PathLike):
        """
        Save the model, abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def load(self, path: PathLike):
        """
        Load the model, abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def serialize(self, param_path: PathLike, serial_path: PathLike):
        """
        Serialize the model, abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def deserialize(self, param_path: PathLike, serial_path: PathLike):
        """
        Deserialize the model, abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the model, abstract method to be implemented by subclasses
        """

    @abstractmethod
    def predict(self, input: Any):
        """
        Predict using the model, abstract method to be implemented by subclasses
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __eq__(self, value):
        # exclude model from comparison
        return self.dict(exclude={"model"}) == value.dict(exclude={"model"})


class PickleableModelBase(ModelBase):
    # classvar for pickleable model
    pickleable: ClassVar[bool] = True

    _model_save_name: ClassVar[str] = "model.pkl"

    def save(self, path: PathLike):
        if self.estimator is None:
            raise ValueError("Model is not built, cannot save")

        with open(path, "wb") as f:
            joblib.dump(self.estimator, f)

    def load(self, path: PathLike):
        with open(path, "rb") as f:
            self.estimator = joblib.load(f)

    def make_new(self) -> "PickleableModelBase":
        """
        Copy parameters to a new model instance without copying the estimator
        """
        return self.__class__(**self.mod_params, **self.dict(exclude={"estimator"}))

    @classmethod
    def deserialize(
        cls, param_path: PathLike = "model.json", serial_path: PathLike = "model.pkl"
    ):
        """
        Create a model from parameters and a pickled model
        """
        with open(param_path) as f:
            mod_params = json.load(f)
        instance = cls(**mod_params)
        instance.build()
        instance.load(serial_path)
        return instance

    def serialize(
        self, param_path: PathLike = "model.json", serial_path: PathLike = "model.pkl"
    ):
        """
        Save the model to a json file and a pickled file
        """
        with open(param_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        self.save(serial_path)


class LightningModuleBase(pl.LightningModule):
    """
    A PyTorch lightning model may inherit this instead of pl.LightningModule
    to preconfigure optimizer and scheduler.

    """

    # Optimizer and scheduler configuration
    optimizer: str = "adamw"
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-5
    scheduler: str = "cosine"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    monitor_metric: str = "val_loss"

    # This must be set for Pydantic to be happy
    # Not certain of reason for this
    training: bool = True

    @field_validator("monitor_metric")
    @classmethod
    def check_monitor_metric(cls, value):
        """
        Check if the monitor metric is valid
        """
        allowed = ["val_loss", "train_loss"]
        if value.lower() not in allowed:
            raise ValueError(f"Monitored metric must be one of {allowed}")
        return value

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, value):
        allowed = {"adamw", "adam", "sgd"}
        if value.lower() not in allowed:
            raise ValueError(f"Optimizer must be one of {allowed}")
        return value

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, value):
        allowed = {"cosine", "reduce_on_plateau", "none", None}
        if (value.lower() not in allowed) and (value is not None):
            raise ValueError(f"Scheduler must be one of {allowed}")
        return value

    def configure_optimizers(self):
        """
        Returns optimizer and scheduler configuration for Lightning's configure_optimizers.
        """
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

    # TODO: Implement defaults of the following?
    # def training_step():
    #     pass

    # def validation_step():
    #     pass

    # def predict_step():
    #     pass


class LightningModelBase(ModelBase):
    _model_save_name: ClassVar[str] = "model.pth"

    def save(self, path: PathLike):
        torch.save(self.estimator.state_dict(), path)

    def load(self, path: PathLike):
        self.estimator.load_state_dict(torch.load(path))

    def serialize(
        self, param_path: PathLike = "model.json", serial_path: PathLike = "model.pth"
    ):
        with open(param_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        self.save(serial_path)

    @classmethod
    def deserialize(
        cls, param_path: PathLike = "model.json", serial_path: PathLike = "model.pth"
    ):
        with open(param_path) as f:
            mod_params = json.load(f)
        instance = cls(**mod_params)
        instance.build()
        instance.load(serial_path)
        return instance
