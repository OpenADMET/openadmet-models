from typing import Any, Iterable, Literal
import numpy as np
import datamol as dm
from pydantic import Field, field_validator
from molfeat.trans import MoleculeTransformer
from openadmet_models.features.base import MolfeatFeaturizer




class DescriptorFeaturizer(MolfeatFeaturizer):
    """
    Fingerprint featurizer for molecules, relies on molfeat backend
    """
    type: Literal["DescriptorFeaturizer"] = "DescriptorFeaturizer"
    descr_type: str = Field(..., title="Descriptor type", description="The type of descriptor to use, must be one of 'mordred', desc2d', 'desc3d'")
    dtype: Any = Field(np.float32, title="Data type", description="The data type to use for the fingerprint")
    n_jobs: int = Field(-1, title="Number of jobs", description="The number of jobs to use for featurization, -1 for maximum parallelism")

    @field_validator("descr_type")
    @classmethod
    def validate_descr_type(cls, value):
        """
        Validate the descriptor type
        """
        if value not in ["mordred", "desc2d", "desc3d"]:
            raise ValueError("Descriptor type must be one of 'mordred', 'desc2d', 'desc3d'")
        return value



    def _prepare(self):
        """
        Prepare the featurizer
        """
        self._transformer = MoleculeTransformer(self.descr_type, n_jobs=self.n_jobs,  dtype=self.dtype, parallel_kwargs = {"progress": False}, verbose=True)


    def featurize(self, smiles: Iterable[str]) -> np.ndarray:
        """
        Featurize a list of SMILES strings
        """
        with dm.without_rdkit_log():
            feat, indices = self._transformer(smiles, ignore_errors=True)
        
        return feat, indices
    



    



