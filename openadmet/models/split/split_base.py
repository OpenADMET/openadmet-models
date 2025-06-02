from abc import ABC, abstractmethod
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

    train_size: float = 0.8
    val_size: float = 0.0
    test_size: float = 0.2
    random_state: int = 42

    @model_validator(mode="after")
    def check_sizes(self):
        # Check that sizes sum to 1
        if self.test_size + self.val_size + self.train_size != 1.0:
            raise ValueError("Test and train sizes must sum to 1.0")

        # Check that val_size and test_size are not both 0
        if self.val_size + self.test_size == 0.0:
            raise ValueError("Either val_size or test_size must be greater than 0")

        return self

    @abstractmethod
    def split(self, X: Iterable, Y: Iterable) -> tuple[Iterable, Iterable]:
        """ """
        pass
