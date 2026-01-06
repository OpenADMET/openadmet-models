"""Base classes and utilities for data splitters."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel, model_validator
from loguru import logger

splitters = ClassRegistry(unique=True)


def get_splitter_class(feat_type):
    """
    Retrieve a splitter class from the registry by type.

    Parameters
    ----------
    feat_type : str
        The type of splitter to retrieve.

    Returns
    -------
    SplitterBase
        The splitter class corresponding to the given type.

    """
    try:
        split_class = splitters.get_class(feat_type)
    except RegistryKeyError:
        raise ValueError(f"Splitter type {feat_type} not found in splitter catalouge")
    return split_class


class SplitterBase(BaseModel, ABC):
    """
    Base class for splitters, allows for arbitrary splitting of data.

    Attributes
    ----------
    train_size : float
        The proportion of the data to use for training, must be between 0 and 1.
    val_size : float
        The proportion of the data to use for validation, must be between 0 and 1.
    test_size : float
        The proportion of the data to use for testing, must be between 0 and 1.
    random_state : int
        The random seed to use for reproducibility.

    """

    train_size: float = 0.8
    val_size: float = 0.0
    test_size: float = 0.2
    random_state: int = 42

    @model_validator(mode="after")
    def check_sizes(self):
        """Validate the sizes of the splits."""
        # Check that sizes sum to 1
        if self.test_size + self.val_size + self.train_size != 1.0:
            raise ValueError("Test and train sizes must sum to 1.0")

        # Check that val_size and test_size are not both 0
        if self.val_size + self.test_size == 0.0:
            logger.info(
                "Warning! val_size and test_size are both 0.0. You are training a no-split model!"
            )

        return self

    @abstractmethod
    def split(self, X: Iterable, Y: Iterable) -> tuple[Iterable, Iterable]:
        """
        Split the data.

        Parameters
        ----------
        X : Iterable
            Feature data.
        Y : Iterable
            Target data.

        Returns
        -------
        tuple
            Tuple containing:
            - X_train: Training set features.
            - X_val: Validation set features (or None if val_size=0).
            - X_test: Test set features (or None if test_size=0).
            - y_train: Training set target values.
            - y_val: Validation set target values (or None if val_size=0).
            - y_test: Test set target values (or None if test_size=0).

        """
        pass
