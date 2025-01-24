from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
from pydantic import BaseModel
from molfeat.trans import MoleculeTransformer




class FeaturizerBase(BaseModel, ABC):
    """
    Base class for featurizers, allows for arbitrary featurization of molecules
    withing the featurize method
    """




    @abstractmethod
    def featurize(self, smiles: Iterable[str]) ->  np.ndarray:
        """
        Featurize a list of SMILES strings
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
        


    