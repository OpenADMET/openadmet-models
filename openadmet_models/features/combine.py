from functools import reduce
from typing import Iterable
import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field
from openadmet_models.features.feature_base import FeaturizerBase
from openadmet_models.features.feature_base import featurizers


@featurizers.register("FeatureConcatenator")
class FeatureConcatenator(FeaturizerBase):
    
    featurizers: list[FeaturizerBase] = Field(
        ..., description="List of featurizers to concatenate"
    )


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
