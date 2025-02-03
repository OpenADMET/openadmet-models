from functools import reduce
from typing import Iterable
import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field, field_validator
from openadmet_models.features.feature_base import FeaturizerBase
from openadmet_models.features.feature_base import featurizers, get_featurizer_class


@featurizers.register("FeatureConcatenator")
class FeatureConcatenator(FeaturizerBase):
    
    featurizers: list[FeaturizerBase] = Field(
        ..., description="List of featurizers to concatenate"
    )


    @field_validator("featurizers", mode='before')
    @classmethod
    def validate_featurizers(cls, value):
        """
        If passed a dictionary of parameters, construct the relevant featurizers
        and pack them into the featurizers list
        """
        print(value)
        if isinstance(value, dict):
            featurizers = []
            for feat_type, feat_params in value.items():
                feat_class = get_featurizer_class(feat_type)
                feat = feat_class(**feat_params)
                featurizers.append(feat)
            return featurizers
        return value


    def featurize(self, smiles: list[str]) -> np.ndarray:
        """
        Featurize a list of SMILES strings
        """
        features = []
        indices = []
        for feat in self.featurizers:
            feat, idx = feat.featurize(smiles)
            features.append(feat)
            indices.append(idx)

        return self.concatenate(features, indices, smiles)


    @staticmethod
    def concatenate(
        feats: list[ArrayLike], indices: list[np.ndarray], smiles: Iterable[str]
    ) -> np.ndarray:
        """
        Concatenate a list of features,
        """

        # use indices to mask out the features that are not present in all datasets
        common_indices = reduce(np.intersect1d, indices)


        # concatenate the features column wise
        concat_feats = np.concatenate(feats, axis=1)
    
        return concat_feats, common_indices, 
