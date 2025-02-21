import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar
import torch
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

    @classmethod
    def from_model_json_and_pkl(
        cls, model_json_path: Pathy = "model.json", pkl_path: Pathy = "model.pkl"
    ):
        """
        Create a model from parameters and a pickled model
        """
        with open(model_json_path) as f:
            model_params = json.load(f)
        instance = cls(**model_params)
        instance.load(pkl_path)
        return instance

    def to_model_json_and_pkl(
        self, model_json_path: Pathy = "model.json", pkl_path: Pathy = "model.pkl"
    ):
        """
        Save the model to a json file and a pickled file
        """
        with open(model_json_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        self.save(pkl_path)

    def serialize_model(self, model_json_path: Pathy = "model.json", pkl_path: Pathy = "model.pkl"):
        return self.to_model_json_and_pkl(model_json_path, pkl_path)
    
    def deserialize_model(self, model_json_path: Pathy = "model.json", pkl_path: Pathy = "model.pkl"):
        return self.from_model_json_and_pkl(model_json_path, pkl_path)


class TorchModelBase(ModelBase):

    def save(self: Any, path: Pathy):
        """
        Save the model
        """
        torch.save(self.model.state_dict(), path)

    def load(self: Any, path: Pathy):
        """
        Load the model
        """
        self._model.load_state_dict(torch.load(path))


    def serialize_model(self, path: Pathy):
        return self.save(path)
    
    def deserialize_model(self, path: Pathy):
        return self.load(path)
