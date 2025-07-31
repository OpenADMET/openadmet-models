import numpy as np
from numpy.typing import ArrayLike
from typing import Literal
from random import sample
from pydantic import Field, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.features.feature_base import DeepLearningFeaturizer

from nepare.data import PairwiseAugmentedDataset

class PairedFeaturizer(DeepLearningFeaturizer):
    """ 
    PairFeaturizedData is a featurizer that pairs features
    according to a specified method
    """
    featurizer: str = Field(
        type=str,
        default="ChemPropFeaturizer",
        description="Featurizer to use before pairing",
    )

    how_to_pair: Literal['all', 'ut', 'sut', 'rand'] = Field(
        "all",
        description="How to pair the features, options are 'all' for all pairs, " \
        "'ut' for upper triangular pairs, 'sut' for symmetric upper triangular pairs,"
        "'rand' for random set of pairs from all, as set by num_pairs.",
    )
    n_jobs: int = 4
    batch_size: int = 128
    shuffle: bool = False

    @model_validator(mode="after")
    def validate_pairwise(self):
        """
        Validate the how_to_pair and num_pairs parameters together.
        """
        if self.how_to_pair not in ["all", "ut", "sut"]:
            raise ValueError(
                "how_to_pair must be one of 'all', 'ut', or 'sut'"
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
        featurizer_cls = globals()[self.featurizer]
        featurizer = featurizer_cls()
        _, _, scaler, unpaired_dataset = featurizer.featurize(smiles, y)

        paired_dataset = PairwiseAugmentedDataset(dataset_features, dataset_labels, how=self.how_to_pair)

        dataloader = DataLoader(
            paired_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_jobs,
        )
        
        return dataloader, paired_dataset.indices
