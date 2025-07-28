import numpy as np
from numpy.typing import ArrayLike
from typing import Literal
from random import sample
from pydantic import Field, field_validator, model_validator
from itertools import combinations, combinations_with_replacement, product, chain

from openadmet.models.features.feature_base import (
    FeaturizerBase,
    featurizers,
)

@featurizers.register("PairFeaturizedData")
class PairFeaturizedData(FeaturizerBase):
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
    num_pairs: int = Field(
        default=None,
        description="Number of pairs to sample if 'rand' is selected for how_to_pair.",
    )

    @model_validator(mode="after")
    def validate_pairwise(self):
        """
        Validate the how_to_pair and num_pairs parameters together.
        """
        if self.how_to_pair not in ["all", "ut", "sut", "rand"]:
            raise ValueError(
                "how_to_pair must be one of 'all', 'ut', 'sut', or 'rand'"
            )
        if self.how_to_pair == "rand" and (not hasattr(self, "num_pairs") or self.num_pairs <= 0):
            raise ValueError(
                "num_pairs must be greater than 0 when how_to_pair is 'rand'"
            )
        return self

    def featurize(self):
        """
        This method is not implemented as PairFeaturizedData is not a featurizer
        but a data processor for pairing features.
        """
        raise NotImplementedError("PairFeaturizedData does not implement featurization.")

    def pair_data(self, x_feat, y) -> np.ndarray:
        """
        Pair the features according to the specified method

        Parameters
        ----------
        x_feat : np.ndarray
            The feature matrix to concatenate.
        x_ind : np.ndarray
            The indices of the features.
        y : np.ndarray
            The y values to compute differences for.

        Returns
        -------
        np.ndarray
            The concatenated feature matrix, and the differences in y values.
        """

        all_pairs = list(product(range(len(y)), repeat=2))
        if self.how_to_pair == "all":
            # Pair all features with all others
            pairs = all_pairs
        elif self.how_to_pair == "ut":
            # Upper triangular pairing
            pairs = list(combinations_with_replacement(range(len(y)), 2))
        elif self.how_to_pair == "sut":
            # Symmetric upper triangular pairing
            pairs = list(combinations(range(len(y)), 2))
        elif self.how_to_pair == "rand":
            # Randomly sample pairs
            if self.num_pairs > len(y):
                raise ValueError("num_pairs exceeds the number of possible pairs.")
            pairs = sample(all_pairs, self.num_pairs)
        else:
            raise ValueError(f"Unknown pairing method: {self.how_to_pair}")

        y_diffs = []
        for pair in pairs:
            idx1, idx2 = pair
            y_diffs.append(y[idx1] - y[idx2])

        x_concat = []
        for pair in pairs:
            idx1, idx2 = pair
            x_concat.append(np.concatenate((x_feat[idx1], x_feat[idx2])))

        return np.array(x_concat), np.array(y_diffs), pairs
