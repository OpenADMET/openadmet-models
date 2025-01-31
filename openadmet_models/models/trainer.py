from abc import ABC, abstractmethod
from typing import Any, ClassVar
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel

trainers = ClassRegistry(unique=True)


def get_trainer_class(model_type):
    try:
        feat_class = trainers.get_class(model_type)
    except RegistryKeyError:
        raise ValueError(f"Trainer type {model_type} not found in trainer catalouge")
    return feat_class


class TrainerBase(BaseModel, ABC):
    model: Any

    @classmethod
    def from_model(cls, model: Any, **kwargs):
        """
        Create a trainer from a model, abstract method to be implemented by subclasses
        """
        return cls(model=model, **kwargs)


    @abstractmethod
    def train(self, X: Any, y: Any):
        """
        Train the model, abstract method to be implemented by subclasses
        """
        pass
  


@trainers.register("SKlearnBasicTrainer")
class SKlearnBasicTrainer(TrainerBase):
    """
    Basic trainer for sklearn models
    """

    type: ClassVar[str] = "SKlearnBasicTrainer"
    

    def train(self, X: Any, y: Any):

        self.model = self.model.fit(X, y)
        return self.model
    

@trainers.register("SKLearnGridSearchTrainer")
class SKLearnGridSearchTrainer(TrainerBase):
    """
    Trainer for sklearn models with grid search
    """

    type: ClassVar[str] = "SKLearnGridSearchTrainer"
    grid_params: dict = {}
    search: Any

    

    def train(self, X: Any, y: Any):
        """
        Train the model
        """
        self.model = GridSearchCV(self.model, param_grid=self.grid_params)
        self.search.fit(X, y)
        self.model = self.search.best_estimator_
        return self.model, self.search.best_params_
    

@trainers.register("SKLearnRandomSearchTrainer")
class SKLearnRandomSearchTrainer(TrainerBase):
    """
    Trainer for sklearn models with random search
    """

    type: ClassVar[str] = "SKLearnRandomSearchTrainer"
    grid_params: dict = {}
    search: Any

    
    def train(self, X: Any, y: Any):
        """
        Train the model
        """
        self.search = RandomizedSearchCV(self.model, grid_params=self.grid_params)
        self.search.fit(X, y)
        self.model = self.search.best_estimator_
        return self.model, self.search.best_params_