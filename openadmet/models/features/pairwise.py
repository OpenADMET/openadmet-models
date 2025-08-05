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


@featurizers.register("NepareChemPropFeaturizer")
class NepareChemPropFeaturizer(DeepLearningFeaturizer):
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
    n_jobs: int = 4
    batch_size: int = 128
    shuffle: bool = False

    @model_validator(mode="after")
    def validate_pairwise(self):
        """
        Validate the how_to_pair and num_pairs parameters together.
        """
        if self.how_to_pair not in ["full", "ut", "sut"]:
            raise ValueError(
                "how_to_pair must be one of 'full', 'ut', or 'sut'"
            )
        return self

    def featurize(
        self,
        smiles: ArrayLike,
        y: ArrayLike = None,
    ) -> tuple[DataLoader, np.ndarray, StandardScaler, Dataset]:
        """
        Featurize a list of SMILES strings. Returns a DataLoader, a list of indices that correspond to the original input, a StandardScaler if any scaling done by featurization, and a Pytorch Dataset
        """
        featurizer = ChemPropFeaturizer(n_jobs=self.n_jobs, normalize_targets=False, batch_size=self.batch_size, shuffle=self.shuffle)
        _, _, _, unpaired_dataset = featurizer.featurize(smiles, y)

        X = np.array(unpaired_dataset.smiles)
        y = np.array([dp.y[0] for dp in unpaired_dataset.data]) # assumes target is a single value

        paired_dataset = PairwiseAugmentedDataset(X, y, how=self.how_to_pair)

        dataloader = DataLoader(
            paired_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_jobs,
        )

        indices = np.arange(len(paired_dataset.X))

        return dataloader, indices, None, paired_dataset # returns None for scaler to stay consistent with ChempropFeaturizer
