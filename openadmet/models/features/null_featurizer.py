"""Null (passthrough) featurizer for lightweight testing."""

from collections.abc import Iterable
from typing import ClassVar

import numpy as np

from openadmet.models.features.feature_base import FeaturizerBase, featurizers


@featurizers.register("NullFeaturizer")
class NullFeaturizer(FeaturizerBase):
    """
    Null featurizer that returns a zero-filled feature matrix.

    Produces a single zero feature per molecule and returns the full index array.
    Useful for unit testing with DummyRegressorModel where feature values are irrelevant.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the featurizer.

    """

    type: ClassVar[str] = "NullFeaturizer"

    def featurize(self, smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Return zero features for all input SMILES.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings.

        Returns
        -------
        tuple
            Tuple of (features, indices). Features is a 2D array of shape (n, 1)
            filled with zeros; indices is a 1D array of integers 0..n-1.

        """
        smiles_list = list(smiles)
        n = len(smiles_list)
        return np.zeros((n, 1), dtype=np.float32), np.arange(n)
