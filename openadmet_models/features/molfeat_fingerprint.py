from collections.abc import Iterable
from typing import Any, ClassVar, Tuple

import datamol as dm
import numpy as np
from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from pydantic import Field

from openadmet_models.features.feature_base import MolfeatFeaturizer, featurizers


@featurizers.register("FingerprintFeaturizer")
class FingerprintFeaturizer(MolfeatFeaturizer):
    """
    Fingerprint featurizer for molecules, relies on molfeat backend
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
        """
        Prepare the featurizer
        """
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
        Featurize a list of SMILES strings
        """
        with dm.without_rdkit_log():
            feat, indices = self._transformer(smiles, ignore_errors=True)
        # datamol returns with an extra dimension
        return np.squeeze(feat), indices
