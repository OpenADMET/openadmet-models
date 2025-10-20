"""Pairwise featurizer implementation."""

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from typing import Literal, Type, Union, Any
from random import sample
from pydantic import Field, field_validator, model_validator
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from collections.abc import Sequence

from collections.abc import Sequence
from collections import namedtuple
from typing import Literal
from itertools import combinations, combinations_with_replacement, product, chain
from random import Random

from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.features.feature_base import DeepLearningFeaturizer, featurizers
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer

from openadmet.models.features.feature_base import (
    FeaturizerBase,
    featurizers,
    get_featurizer_class,
)


class PairwiseAugmentedDataset(torch.utils.data.Dataset):
    """
    Subclass of PairwiseAugmentedDataset to handle inference cases where y is None.

    Based on: https://github.com/JacksonBurns/neural-pairwise-regression/blob/main/nepare/data.py

    """

    def __init__(
        self, X: Sequence, y: Sequence, *, how: Literal["full", "ut", "sut"] = "full"
    ):
        """Initialize the PairwiseAugmentedDataset."""
        super().__init__()
        self.X = X
        self.y = y
        match how:
            case "full":
                self.idxs = list(product(range(len(X)), repeat=2))
            case "ut":
                self.idxs = list(combinations_with_replacement(range(len(X)), 2))
            case "sut":
                self.idxs = list(combinations(range(len(X)), 2))
            case _:
                raise TypeError(f"Invalid configuration {how=}.")

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.idxs)

    def __getitem__(self, index):
        """Return the item at the given index."""
        i, j = self.idxs[index]
        if self.y is not None:
            return self.X[i], self.X[j], self.y[i] - self.y[j]
        else:
            return self.X[i], self.X[j]

    def downsample_(self, n: int, random_seed: int = 1701):
        """Downsample the dataset to n pairs."""
        rng = Random(random_seed)
        self.idxs = rng.sample(self.idxs, k=n)


@featurizers.register("PairwiseFeaturizer")
class PairwiseFeaturizer(FeaturizerBase):
    """
    PairFeaturizedData is a featurizer that pairs features according to a specified method.

    Attributes
    ----------
    how_to_pair : str
        Method to pair features. Options are 'full' for all pairs, 'ut' for
        upper triangular pairs, 'sut' for symmetric upper triangular pairs.
    featurizer : Union[type[FeaturizerBase], FeaturizerBase, dict
        Featurizer to use before pairing. Can be a FeaturizerBase subclass,
        an instance of a FeaturizerBase subclass, or a dictionary of parameters
        to construct a FeaturizerBase subclass.
    n_jobs : int
        Number of jobs to use for featurization.
    batch_size : int
        Batch size to use for DataLoader.
    shuffle : bool
        Whether to shuffle the data in the DataLoader.

    """

    how_to_pair: Literal["full", "ut", "sut"] = "full"
    featurizer: FeaturizerBase = FingerprintFeaturizer(fp_type="ecfp:4")
    n_jobs: int = 4
    batch_size: int = 128
    shuffle: bool = False

    @field_validator("how_to_pair", mode="before")
    @classmethod
    def validate_pairwise(cls, value):
        """Validate the how_to_pair parameter."""
        if value not in ["full", "ut", "sut"]:
            raise ValueError("how_to_pair must be one of 'full', 'ut', or 'sut'")
        return value

    @field_validator("featurizer", mode="before")
    @classmethod
    def validate_featurizer(cls, value):
        """If passed a dictionary of parameters, construct the relevant featurizer and return it."""
        if isinstance(value, dict):
            if len(value) != 1:
                raise ValueError(
                    "Only a single featurizer can be specified for 'featurizer'. "
                    f"Received: {list(value.keys())}"
                )
            feat_type, feat_params = next(iter(value.items()))
            feat_class = get_featurizer_class(feat_type)
            return feat_class(**feat_params)
        elif isinstance(value, FeaturizerBase):
            return value
        else:
            raise TypeError(
                "Input should be a valid dictionary of featurizer parameters or a FeaturizerBase subclass"
                f" [type=model_type, input_value={value}, input_type={type(value)}]"
            )

    def featurize(
        self,
        smiles: ArrayLike,
        y: ArrayLike = None,
    ) -> tuple[DataLoader, np.ndarray, StandardScaler, Dataset]:
        """
        Featurize a list of SMILES strings.

        Returns a DataLoader, a list of indices that correspond to the original input, a StandardScaler if any scaling done by featurization, and a Pytorch Dataset

        Parameters
        ----------
        smiles: ArrayLike
            A list or array of SMILES strings to featurize
        y: ArrayLike, optional
            A list or array of target values to pair with the features

        Returns
        -------
        dataloader: DataLoader
            A DataLoader containing the paired features and targets
        indices: np.ndarray
            An array of indices that correspond to the original input
        scaler: StandardScaler or None
            A StandardScaler if any scaling done by featurization, else None
        dataset: Dataset
            A Pytorch Dataset containing the paired features and targets

        """
        X_feat, _ = self.featurizer.featurize(smiles)
        X_feat = X_feat.astype(np.float32)
        if y is not None:
            # Convert y to 1D numpy array if it's a DataFrame or Series
            if isinstance(y, pd.DataFrame):
                y = y.values.ravel()
            elif isinstance(y, pd.Series):
                y = y.values
            else:
                y = np.asarray(y)
            y = y.astype(np.float32)

        paired_dataset = PairwiseAugmentedDataset(X_feat, y, how=self.how_to_pair)

        dataloader = DataLoader(
            paired_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_jobs,
        )

        indices = np.arange(len(paired_dataset.X))

        return dataloader, indices, None, paired_dataset

    def make_new(self) -> "PairwiseFeaturizer":
        """
        Copy parameters to a new PairwiseFeaturizer instance, ensuring the featurizer field is in the correct format.

        Returns
        -------
        PairwiseFeaturizer
            A new instance of PairwiseFeaturizer with the same parameters.

        """
        # Get the featurizer type name
        ft = PairwiseFeaturizer(
            how_to_pair=self.how_to_pair,
            featurizer=self.featurizer,
            n_jobs=self.n_jobs,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        return ft
