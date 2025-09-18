"""Base class for trainers, allows for arbitrary training of models."""

from abc import ABC, abstractmethod
from typing import Any

from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel

from openadmet.models.architecture.model_base import ModelBase

trainers = ClassRegistry(unique=True)


def get_trainer_class(model_type):
    """
    Retrieve a trainer class from the registry by type.

    Parameters
    ----------
    model_type : str
        The type of trainer to retrieve.

    Returns
    -------
    TrainerBase
        The trainer class corresponding to the given type.

    """
    try:
        feat_class = trainers.get_class(model_type)
    except RegistryKeyError:
        raise ValueError(
            f"Trainer type {model_type} not found in trainer catalouge,"
            f"available trainers are {list(trainers.classes())}"
        )
    return feat_class


class TrainerBase(BaseModel, ABC):
    """
    Base class for trainers, allows for arbitrary training of models.

    Attributes
    ----------
    _model : ModelBase
        The model to be trained.

    """

    _model: ModelBase

    @property
    def model(self):
        """Return model to be trained."""
        return self._model

    @model.setter
    def model(self, model):
        """Set model to be trained."""
        self._model = model

    @abstractmethod
    def build():
        """Build trainer, to be implemented by subclasses."""
        pass

    @abstractmethod
    def train(self, X: Any, y: Any):
        """
        Train the model, abstract method to be implemented by subclasses.

        Parameters
        ----------
        X : Any
            Feature data.
        y : Any
            Target data.

        """
        pass
