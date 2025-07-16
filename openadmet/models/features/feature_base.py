from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from class_registry import ClassRegistry, RegistryKeyError
from molfeat.trans import MoleculeTransformer
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

featurizers = ClassRegistry(unique=True)


def get_featurizer_class(feat_type):
    try:
        feat_class = featurizers.get_class(feat_type)
    except RegistryKeyError:
        raise ValueError(f"Feature type {feat_type} not found in feature catalouge")
    return feat_class


class FeaturizerBase(BaseModel, ABC):
    """
    Base class for featurizers, allows for arbitrary featurization of molecules
    withing the featurize method
    """

    @abstractmethod
    def featurize(self, smiles: Iterable[str], *args, **kwargs):
        """
        Featurize a list of SMILES strings, returns features in an appropriate format
        for the model (numpy arrays, dataloaders, etc.) and optional processing info
        """


class DeepLearningFeaturizer(FeaturizerBase):
    """
    Base class for deep learning featurizers, allows for arbitrary featurization of molecules
    withing the featurize method
    """

    @abstractmethod
    def featurize(
        self, smiles: Iterable[str], y: Iterable[float] = None
    ) -> tuple[DataLoader, np.ndarray, StandardScaler, Dataset]:
        """
        Featurize a list of SMILES strings. Returns a DataLoader, a list of indices that correspond to the original input, a StandardScaler if any scaling done by featurization, and a Pytorch Dataset
        """


class MolfeatFeaturizer(FeaturizerBase):
    """
    Featurizer using molfeat
    """

    _transformer: MoleculeTransformer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare()

    @abstractmethod
    def _prepare(self):
        """
        Prepare the featurizer
        """

    @property
    def transformer(self):
        """
        Return the transformer, for use in SkLearn pipelines etc
        """
        return self._transformer
