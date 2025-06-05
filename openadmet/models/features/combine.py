from functools import reduce

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field, field_validator

from openadmet.models.features.feature_base import (
    FeaturizerBase,
    featurizers,
    get_featurizer_class,
)


@featurizers.register("FeatureConcatenator")
class FeatureConcatenator(FeaturizerBase):
    featurizers: list[FeaturizerBase] = Field(
        ..., description="List of featurizers to concatenate"
    )

    @field_validator("featurizers", mode="before")
    @classmethod
    def validate_featurizers(cls, value):
        """
        If passed a dictionary of parameters, construct the relevant featurizers
        and pack them into the featurizers list
        """
        processed_featurizers = []
        if isinstance(value, dict):
            for feat_type, feat_params in value.items():
                feat_class = get_featurizer_class(feat_type)
                feat = feat_class(**feat_params)
                processed_featurizers.append(feat)
        elif isinstance(value, list):
            processed_featurizers = value
        else:
            # Or raise an error if the type is unexpected
            return value

        # Sort the featurizers by class name
        return sorted(processed_featurizers, key=lambda f: f.__class__.__name__)

    def featurize(self, smiles: list[str]) -> np.ndarray:
        """
        Featurize a list of SMILES strings
        """
        features = []
        indices = []
        for feat in self.featurizers:
            feat_res, idx = feat.featurize(smiles)
            features.append(feat_res)
            indices.append(idx)

        return self.concatenate(features, indices)

    @staticmethod
    def concatenate(feats: list[ArrayLike], indices: list[np.ndarray]) -> np.ndarray:
        """
        Concatenate a list of features,
        """

        # if the input arrays are 1d, make them 2d
        feats = [
            feat.reshape(1, -1) if len(feat.shape) == 1 else feat for feat in feats
        ]

        # use indices to mask out the features that are not present in all datasets
        common_indices = reduce(np.intersect1d, indices)

        # handle 1d features from single input by making them 2d
        # concatenate the features column wise
        concat_feats = np.concatenate(feats, axis=1)
        return (
            concat_feats,
            common_indices,
        )
