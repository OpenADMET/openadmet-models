from abc import ABC, abstractmethod
from typing import Any

from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel

from openadmet.models.architecture.model_base import ModelBase

trainers = ClassRegistry(unique=True)


def get_trainer_class(model_type):
    try:
        feat_class = trainers.get_class(model_type)
    except RegistryKeyError:
        raise ValueError(
            f"Trainer type {model_type} not found in trainer catalouge,"
            f"available trainers are {list(trainers.classes())}"
        )
    return feat_class


class TrainerBase(BaseModel, ABC):
    _model: ModelBase

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @abstractmethod
    def train(self, X: Any, y: Any):
        """
        Train the model, abstract method to be implemented by subclasses
        """
        pass
