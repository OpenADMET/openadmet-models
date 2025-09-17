"""Fingerprint featurizer using molfeat library."""

from collections.abc import Iterable
from typing import Any, ClassVar

import datamol as dm
import numpy as np
from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from pydantic import Field

from openadmet.models.features.feature_base import MolfeatFeaturizer, featurizers


@featurizers.register("FingerprintFeaturizer")
class FingerprintFeaturizer(MolfeatFeaturizer):
    """
    Fingerprint featurizer for molecules, relies on molfeat backend.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the featurizer.
    fp_type : str
        The type of fingerprint to use (e.g., 'ecfp4', 'morgan', 'rdkit', etc.).
    dtype : Any
        The data type to use for the fingerprint (e.g., np.float32).
    n_jobs : int
        The number of jobs to use for featurization, -1 for maximum parallelism.

    """

    type: ClassVar[str] = "FingerprintFeaturizer"
    fp_type: str = Field(
        ..., title="Fingerprint type", description="The type of fingerprint to use"
    )
    dtype: Any = Field(
        np.float32,
        title="Data type",
        description="The data type to use for the fingerprint",
    )
    n_jobs: int = Field(
        -1,
        title="Number of jobs",
        description="The number of jobs to use for featurization, -1 for maximum parallelism",
    )

    def _prepare(self):
        """Prepare the featurizer."""
        vec_featurizer = FPVecTransformer(self.fp_type, dtype=self.dtype)
        self._transformer = MoleculeTransformer(
            vec_featurizer,
            n_jobs=self.n_jobs,
            dtype=self.dtype,
            parallel_kwargs={"progress": False},
            verbose=True,
        )

    def featurize(self, smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Featurize a list of SMILES strings.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings to featurize.

        Returns
        -------
        tuple
            Tuple of (features, indices). Features is a 2D numpy array of shape (
            n_samples, n_features) and indices is a 1D numpy array of the indices of the
            successfully featurized molecules.

        """
        with dm.without_rdkit_log():
            feat, indices = self._transformer(smiles, ignore_errors=True)
        # datamol returns with an extra dimension
        return np.squeeze(feat), indices
