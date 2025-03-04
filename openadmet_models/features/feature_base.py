from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from class_registry import ClassRegistry, RegistryKeyError
from molfeat.trans import MoleculeTransformer
from pydantic import BaseModel

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
    def featurize(self, smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Featurize a list of SMILES strings, returns a numpy array of features,
        and a list of indices that correspond to the original input and the input indexed by the indices
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





