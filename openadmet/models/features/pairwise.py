import numpy as np
from numpy.typing import ArrayLike
from typing import Literal
from random import sample
from pydantic import Field, field_validator, model_validator
from itertools import combinations, combinations_with_replacement, product, chain
from torch.utils.data import DataLoader

from openadmet.models.features.feature_base import DeepLearningFeaturizer

class PairedFeaturizer(DeepLearningFeaturizer):
    """ 
    PairFeaturizedData is a featurizer that pairs features
    according to a specified method
    """

    how_to_pair: Literal['all', 'ut', 'sut', 'rand'] = Field(
        "all",
        description="How to pair the features, options are 'all' for all pairs, " \
        "'ut' for upper triangular pairs, 'sut' for symmetric upper triangular pairs,"
        "'rand' for random set of pairs from all, as set by num_pairs.",
    )

    @model_validator(mode="after")
    def validate_pairwise(self):
        """
        Validate the how_to_pair and num_pairs parameters together.
        """
        if self.how_to_pair not in ["all", "ut", "sut"]:
            raise ValueError(
                "how_to_pair must be one of 'all', 'ut', or 'sut'"
            )

    def featurize(
        self,
        smiles: ArrayLike,
        y: ArrayLike = None,
    ) -> tuple[DataLoader, np.ndarray, StandardScaler, Dataset]:
        """
        Featurize a list of SMILES strings. Returns a DataLoader, a list of indices that correspond to the original input, a StandardScaler if any scaling done by featurization, and a Pytorch Dataset
        """
        # Featurization logic here
        # This is a placeholder for the actual implementation
        pass