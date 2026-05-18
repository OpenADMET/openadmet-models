"""Base classes for all models."""

import json
from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, ClassVar

from class_registry import ClassRegistry, RegistryKeyError
from loguru import logger
from pydantic import BaseModel

from openadmet.models.drivers import DriverType

models = ClassRegistry(unique=True)


def get_mod_class(model_type):
    """Get the model class from the registry."""
    from openadmet.models._registry_loader import load_group

    load_group("models")
    try:
        feat_class = models.get_class(model_type)
    except RegistryKeyError:
        raise ValueError(f"Model type {model_type} not found in model catalouge")
    return feat_class


class ModelBase(BaseModel, ABC):
    """Base class for all models."""

    _estimator: Any = None
    _model_json_name: ClassVar[str] = "model.json"
    _n_tasks: int = 1

    @property
    def estimator(self):
        """Get the model estimator."""
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        """Set the model estimator."""
        self._estimator = value

    @abstractmethod
    def build(self):
        """Prepare the model, abstract method to be implemented by subclasses."""
        pass

    @abstractmethod
    def save(self, path: PathLike):
        """
        Save the model, abstract method to be implemented by subclasses.

        Parameters
        ----------
        path: PathLike
            Path to save the model to

        """
        pass

    @abstractmethod
    def load(self, path: PathLike):
        """
        Load the model, abstract method to be implemented by subclasses.

        Parameters
        ----------
        path: PathLike
            Path to load the model from

        """
        pass

    @abstractmethod
    def serialize(self, param_path: PathLike, serial_path: PathLike):
        """
        Serialize the model, abstract method to be implemented by subclasses.

        Parameters
        ----------
        param_path: PathLike
            Path to save the model parameters to
        serial_path: PathLike
            Path to save the model serialization to

        """
        pass

    @abstractmethod
    def deserialize(self, param_path: PathLike, serial_path: PathLike):
        """
        Deserialize the model, abstract method to be implemented by subclasses.

        Parameters
        ----------
        param_path: PathLike
            Path to load the model parameters from
        serial_path: PathLike
            Path to load the model serialization from

        """
        pass

    @abstractmethod
    def train(self):
        """Train the model, abstract method to be implemented by subclasses."""

    @abstractmethod
    def predict(self, input: Any):
        """
        Predict using the model, abstract method to be implemented by subclasses.

        Parameters
        ----------
        input: Any
            Input data to predict on

        """
        pass

    def __call__(self, *args, **kwargs):
        """Call the predict method when the model instance is called."""
        return self.predict(*args, **kwargs)

    def __eq__(self, value):
        """Compare two model instances for equality, ignoring the model itself."""
        # exclude model from comparison
        return self.model_dump(exclude={"estimator"}) == value.model_dump(
            exclude={"estimator"}
        )


class PickleableModelBase(ModelBase):
    """An sklearn model that can be pickled using joblib."""

    # ClassVar for pickleable model
    pickleable: ClassVar[bool] = True
    _model_save_name: ClassVar[str] = "model.pkl"
    _driver_type: DriverType = DriverType.SKLEARN

    def save(self, path: PathLike):
        """
        Save the model to a pickle file.

        Parameters
        ----------
        path: PathLike
            Path to save the model to

        """
        if self.estimator is None:
            raise ValueError("Model is not built, cannot save")

        import joblib

        with open(path, "wb") as f:
            joblib.dump(self.estimator, f)

    def load(self, path: PathLike):
        """
        Load the model from a pickle file.

        Parameters
        ----------
        path: PathLike
            Path to load the model from

        """
        import joblib

        with open(path, "rb") as f:
            self.estimator = joblib.load(f)

    def make_new(self) -> "PickleableModelBase":
        """Copy parameters to a new model instance without copying the estimator."""
        return self.__class__(**self.model_dump(exclude={"estimator"}))

    @classmethod
    def deserialize(
        cls, param_path: PathLike = "model.json", serial_path: PathLike = "model.pkl"
    ):
        """
        Create a model from parameters and a pickled model.

        Parameters
        ----------
        param_path: PathLike
            Path to load the model parameters from
        serial_path: PathLike
            Path to load the pickled model from

        Returns
        -------
        instance: PickleableModelBase
            An instance of the PickleableModelBase class

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
        Save the model to a json file and a pickled file.

        Parameters
        ----------
        param_path: PathLike
            Path to save the model parameters to
        serial_path: PathLike
            Path to save the pickled model to

        """
        with open(param_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        self.save(serial_path)


# Re-export Lightning base classes using lazy module __getattr__ (PEP 562) so that
# importing model_base does NOT pull in torch or lightning.pytorch.
# The actual definitions live in lightning_model_base.
_LIGHTNING_EXPORTS = frozenset({"LightningModelBase", "LightningModuleBase"})


def __getattr__(name: str):
    """Lazily re-export Lightning base classes to avoid paying their import cost."""
    if name in _LIGHTNING_EXPORTS:
        from openadmet.models.architecture import lightning_model_base as _lmb

        value = getattr(_lmb, name)
        # Cache in module dict so subsequent accesses are direct
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelBase",
    "PickleableModelBase",
    "LightningModuleBase",
    "LightningModelBase",
    "models",
    "get_mod_class",
]
