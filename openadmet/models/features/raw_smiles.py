"""Featurizer that preserves raw SMILES strings for downstream models."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from openadmet.models.features.feature_base import FeaturizerBase, featurizers


@featurizers.register("RawSmilesFeaturizer")
class RawSmilesFeaturizer(FeaturizerBase):
    """
    Return SMILES strings without feature transformation.

    This featurizer is intended for model backends that operate directly on
    SMILES strings (e.g., external sequence/graph toolchains) and therefore do
    not require descriptor or fingerprint preprocessing.
    """

    def featurize(self, smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Return raw SMILES as an object array plus positional indices.

        Parameters
        ----------
        smiles : Iterable[str]
            Input SMILES values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Raw SMILES and a positional index array.

        """
        if isinstance(smiles, pd.Series):
            smiles = smiles.to_numpy()

        smiles_array = np.asarray(list(smiles), dtype=object)
        indices = np.arange(len(smiles_array))
        return smiles_array, indices
