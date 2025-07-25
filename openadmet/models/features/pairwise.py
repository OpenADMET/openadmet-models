import numpy as np
from numpy.typing import ArrayLike
from typing import Literal
from pydantic import Field, field_validator

from openadmet.models.features.feature_base import (
    FeaturizerBase,
    featurizers,
    get_featurizer_class,
)

@featurizers.register("PairwiseFeatureizer")
class PairwiseFeatureizer(FeaturizerBase):
    featurizer: FeaturizerBase = Field(
        ..., description="Featurized data to concatenate, pairwise"
    )
    how_to_pair: Literal['all', 'ut', 'sut'] = Field(
        "all",
        description="How to pair the features, options: TODO!!!",
    )

    @field_validator("featurizer", mode="before")
    @classmethod
    def validate_featurizer(cls, value):
        """
        If passed a dictionary of parameters, construct the relevant featurizers
        and pack them into the featurizers list
        """
        if isinstance(value, dict):
            for feat_type, feat_params in value.items():
                feat_class = get_featurizer_class(feat_type)
                feat = feat_class(**feat_params)
        elif isinstance(value, list):
            feat = value
        else:
            # Or raise an error if the type is unexpected
            return value

        # Sort the featurizers by class name
        return feat

    def featurize(self, data: ArrayLike) -> np.ndarray:
        """
        Featurize the input data
        """
        indices, features = self.featurizer.featurize(data)
        return self.pair_data(features, indices)

    def pair_data(self, features: np.ndarray) -> np.ndarray:
        """
        Pair the features according to the specified method
        """
        pass
    # This should happen actually before the featurization,
    # also probably before the split, because this doesn't
    # won't work on the Xs.  But, possibly need to do indexing
    # on the Xs, pass those here, because don't want to featurize
    # on concatenated smiles strings.  So, this still needs to
    # happen, need to add some stuff in the anvil workflow as well.
