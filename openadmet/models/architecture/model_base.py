import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import joblib
import torch
from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel
from os import PathLike


models = ClassRegistry(unique=True)


def get_model_class(model_type):
    try:
        feat_class = models.get_class(model_type)
    except RegistryKeyError:
        raise ValueError(f"Model type {model_type} not found in model catalouge")
    return feat_class


class ModelBase(BaseModel, ABC):
    _estimator: Any = None

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

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

    def save(self, path: PathLike):

        if self.estimator is None:
            raise ValueError("Model is not built, cannot save")

        with open(path, "wb") as f:
            joblib.dump(self.estimator, f)

    def load(self, path: PathLike):

        with open(path, "rb") as f:
            self._estimator = joblib.load(f)

    @classmethod
    def deserialize(
        cls, param_path: PathLike = "model.json", serial_path: PathLike = "model.pkl"
    ):
        """
        Create a model from parameters and a pickled model
        """
        with open(param_path) as f:
            model_params = json.load(f)
        instance = cls(**model_params)
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


class TorchModelBase(ModelBase):
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

    def deserialize(
        self, param_path: PathLike = "model.json", serial_path: PathLike = "model.pth"
    ):
        with open(param_path) as f:
            model_params = json.load(f)
        instance = self.from_params(**model_params)
        instance.load(serial_path)
        return instance
