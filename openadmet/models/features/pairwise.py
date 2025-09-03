import numpy as np
from numpy.typing import ArrayLike
from typing import Literal
from random import sample
from pydantic import Field, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.features.feature_base import DeepLearningFeaturizer, featurizers

from nepare.data import PairwiseAugmentedDataset

from openadmet.models.features.feature_base import (
    FeaturizerBase,
    featurizers,
    get_featurizer_class,
)

@featurizers.register("PairwiseFeaturizer")
class PairwiseFeaturizer(FeaturizerBase):
    """
    PairFeaturizedData is a featurizer that pairs features
    according to a specified method
    """

    how_to_pair: Literal['full', 'ut', 'sut'] = Field(
        "full",
        description="How to pair the features, options are 'full' for all pairs, " \
        "'ut' for upper triangular pairs, 'sut' for symmetric upper triangular pairs,"
        "'rand' for random set of pairs from full, as set by num_pairs.",
    )
    featurizer: FeaturizerBase = Field(
        ..., description="List of featurizers to use before pairing"
    )
    n_jobs: int = Field(
        4, description="Number of jobs to use for featurization"
    )
    batch_size: int = Field(
        128, description="Batch size to use for DataLoader"
    )
    shuffle: bool = Field(
        False, description="Whether to shuffle the data in the DataLoader"
    )

    @model_validator(mode="before")
    def validate_pairwise(cls, values):
        """
        Validate the how_to_pair and num_pairs parameters together.
        """
        how_to_pair = values.get("how_to_pair")
        if how_to_pair not in ["full", "ut", "sut"]:
            raise ValueError(
                "how_to_pair must be one of 'full', 'ut', or 'sut'"
            )
        return values

    @field_validator("featurizer", mode="before")
    @classmethod
    def validate_featurizer(cls, value):
        """
        If passed a dictionary of parameters, construct the relevant featurizer
        and return it
        """
        if isinstance(value, dict):
            for feat_type, feat_params in value.items():
                feat_class = get_featurizer_class(feat_type)
                return feat_class(**feat_params)
        else:
            raise TypeError(
                "Input should be a valid dictionary or instance of FeaturizerBase"
                f" [type=model_type, input_value={value}, input_type={type(value)}]"
            )

    def featurize(
        self,
        smiles: ArrayLike,
        y: ArrayLike = None,
    ) -> tuple[DataLoader, np.ndarray, StandardScaler, Dataset]:
        """
        Featurize a list of SMILES strings. Returns a DataLoader, a list of indices that correspond to the original input, a StandardScaler if any scaling done by featurization, and a Pytorch Dataset
        """

        X_feat, _ = self.featurizer.featurize(smiles)

        paired_dataset = PairwiseAugmentedDataset(X_feat, y, how=self.how_to_pair)

        dataloader = DataLoader(
            paired_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_jobs,
        )

        indices = np.arange(len(paired_dataset.X))

        return dataloader, indices, None, paired_dataset

