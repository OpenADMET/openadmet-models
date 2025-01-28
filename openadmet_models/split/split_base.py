from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
from collections.abc import Iterable

from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel, model_validator

splitters = ClassRegistry(unique=True)


def get_splitter_class(feat_type):
    try:
        split_class = splitters.get_class(feat_type)
    except RegistryKeyError:
        raise ValueError(f"Splitter type {feat_type} not found in splitter catalouge")
    return split_class


class SplitterBase(BaseModel, ABC):
    """
    Base class for splitters, allows for arbitrary splitting of data
    """

    test_size: float = 0.75
    train_size: float = 0.25
    random_state: int = 42

    @model_validator(mode="after")
    def check_sizes(self):
        if self.test_size + self.train_size != 1.0:
            raise ValueError("Test and train sizes must sum to 1.0")
        return self

    @abstractmethod
    def split(self, X: Iterable, Y: Iterable) -> tuple[Iterable, Iterable]:
        """ """
        pass
