"""Base class for transforms, allows for arbitrary transformation of input data."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from class_registry import ClassRegistry, RegistryKeyError
from molfeat.trans import MoleculeTransformer
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

transforms = ClassRegistry(unique=True)


def get_transform_class(trans_type):
    """
    Retrieve a transform class from the registry by type.

    Parameters
    ----------
    trans_type : str
        The type of transform to retrieve.

    Returns
    -------
    TransformBase
        The transform class corresponding to the given type.

    """
    try:
        transf_class = transforms.get_class(trans_type)
    except RegistryKeyError:
        raise ValueError(
            f"Transform type {trans_type} not found in transform catalogue"
        )
    return transf_class


class TransformBase(BaseModel, ABC):
    """Base class for featurizers, allows for arbitrary featurization of molecules."""

    @abstractmethod
    def transform(self, X: np.ndarray, *args, **kwargs):
        """
        Transform the input data X, returns transformed data in an appropriate format.

        Parameters
        ----------
        X : np.ndarray
            Input data to be transformed.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Any
            Transformed data in an appropriate format for the model (e.g., numpy arrays, dataloaders, etc.)
            and optional processing info.

        """
