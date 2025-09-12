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
    try:
        transf_class = transforms.get_class(trans_type)
    except RegistryKeyError:
        raise ValueError(
            f"Transform type {trans_type} not found in transform catalogue"
        )
    return transf_class


class TransformBase(BaseModel, ABC):
    """
    Base class for featurizers, allows for arbitrary featurization of molecules
    withing the featurize method
    """

    @abstractmethod
    def transform(self, X: np.ndarray, *args, **kwargs):
        """
        Transform the input data X, returns transformed data in an appropriate format
        """
