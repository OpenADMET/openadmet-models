"""CheMeleon embedding featurizer."""

from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch

from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.features.feature_base import FeaturizerBase, featurizers


# Foundation checkpoint name; tests monkeypatch this to the hermetic
# chemeleon-test architecture so no checkpoint download is needed
_FOUNDATION_NAME = "chemeleon"

# CheMeleon foundation checkpoint width; used only for the zero-row
# return shape so an empty input never triggers a checkpoint download
_FOUNDATION_EMBEDDING_DIM = 2048


def _normalize_accelerator(accelerator: str) -> str:
    if accelerator == "gpu":
        return "cuda"
    return accelerator


@featurizers.register("CheMeleonEmbeddingFeaturizer")
class CheMeleonEmbeddingFeaturizer(FeaturizerBase):
    """
    Return 2048-length CheMeleon MPNN embeddings for SMILES.

    The featurizer builds a ChemPropModel with the CheMeleon foundation checkpoint
    and extracts pre-predictor pooled embeddings via predict_embedding. No training
    is performed; the pretrained encoder weights are used as-is.

    Parameters
    ----------
    accelerator : str
        Device to use for inference, cpu or cuda.
    batch_size : int
        Number of molecules per forward pass.

    """

    type: ClassVar[str] = "CheMeleonEmbeddingFeaturizer"

    accelerator: str = "cpu"
    batch_size: int = 256

    def __init__(self, accelerator: str = "cpu", batch_size: int = 256):
        """
        Initialize the featurizer with device and batching settings.

        Parameters
        ----------
        accelerator : str, optional
            Device to use for inference, cpu or cuda. Default is cpu.
        batch_size : int, optional
            Number of molecules per forward pass. Default is 256.

        """
        super().__init__()
        self.accelerator = accelerator
        self.batch_size = batch_size
        self._model: ChemPropModel | None = None

    def _ensure_model(self) -> ChemPropModel:
        if self._model is None:
            model = ChemPropModel(from_foundation=_FOUNDATION_NAME)
            model.build()
            device = torch.device(_normalize_accelerator(self.accelerator))
            model.estimator.to(device)
            self._model = model
        return self._model

    def featurize(self, smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Featurize a list of SMILES strings.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings to featurize. Inputs must be
            valid, parsable SMILES; unparsable entries raise an error from
            the underlying toolkit.

        Returns
        -------
        tuple
            Tuple of (features, indices). Features is a 2D numpy array of shape
            (n_samples, embedding_dim) and indices is a 1D numpy array giving
            the input position of each feature row.

        """
        smiles_list = list(smiles)
        if not smiles_list:
            # Width comes from the checkpoint constant so an empty input
            # returns a correctly shaped array without triggering a download
            return (
                np.empty((0, _FOUNDATION_EMBEDDING_DIM), dtype=np.float32),
                np.empty(0, dtype=int),
            )
        model = self._ensure_model()
        # The featurizer decides the device, so hand it to predict_embedding
        # rather than letting it default
        embeddings = model.predict_embedding(
            smiles_list, batch_size=self.batch_size, accelerator=self.accelerator
        )
        # Every input row is featurized, so rows map 1:1 to input positions
        return embeddings, np.arange(len(smiles_list), dtype=int)
