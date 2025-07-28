from abc import ABC, abstractmethod
from collections.abc import Iterable
from pydantic import BaseModel
from class_registry import ClassRegistry, RegistryKeyError


augments = ClassRegistry(unique=True)

def get_augment_class(augment_type):
    try:
        feat_class = augments.get_class(augment_type)
    except RegistryKeyError:
        raise ValueError(f"Feature type {augment_type} not found in feature catalouge")
    return feat_class

class AugmentBase(BaseModel, ABC):
    """
    Base class for Augmenters, allows for arbitrary adjustments to 
    data after featurization
    """

    @abstractmethod
    def augment_data(self, smiles: Iterable[str], *args, **kwargs):
        """
        Augment a list of SMILES strings, returns features in an appropriate format
        for the model, such as pairing featurized data
        """
        pass
