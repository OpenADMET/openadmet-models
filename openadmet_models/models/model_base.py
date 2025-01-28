from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

import joblib
from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel

from openadmet_models.util.types import Pathy

models = ClassRegistry(unique=True)


def get_model_class(model_type):
    try:
        feat_class = models.get_class(model_type)
    except RegistryKeyError:
        raise ValueError(f"Model type {model_type} not found in model catalouge")
    return feat_class


class ModelBase(BaseModel, ABC):
    _model: Any = None
    _built: bool = False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @abstractmethod
    def from_params(cls, class_params: dict, model_params: dict):
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
    def save(self, path: Pathy):
        """
        Save the model, abstract method to be implemented by subclasses
        """
        pass

    @abstractmethod
    def load(self, path: Pathy):
        """
        Load the model, abstract method to be implemented by subclasses
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

    def save(self, path: Pathy):

        if self.model is None:
            raise ValueError("Model is not built, cannot save")

        with open(path, "wb") as f:
            joblib.dump(self.model, f)

    def load(self, path: Pathy):

        with open(path, "rb") as f:
            self._model = joblib.load(f)
