"""CheMeleon embedding featurizer."""

from collections.abc import Iterable
from typing import ClassVar

import datamol as dm
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
            List or iterable of SMILES strings to featurize. Unparseable
            SMILES are skipped, so embeddings may be fewer rows than inputs.

        Returns
        -------
        tuple
            Tuple of (features, indices). Features is a 2D numpy array of shape
            (n_samples, embedding_dim) and indices is a 1D numpy array of the
            indices of the successfully featurized molecules.

        """
        smiles_list = list(smiles)
        # Downstream splits and alignment work on the returned indices, so every
        # returned row must correspond to a parseable input; unparseable
        # entries are skipped, not fatal
        with dm.without_rdkit_log():
            valid = [i for i, s in enumerate(smiles_list) if dm.to_mol(s) is not None]
        if not valid:
            return (
                np.empty((0, _FOUNDATION_EMBEDDING_DIM), dtype=np.float32),
                np.empty(0, dtype=int),
            )
        model = self._ensure_model()
        selected = [smiles_list[i] for i in valid]
        embeddings = model.predict_embedding(selected, batch_size=self.batch_size)
        return embeddings, np.asarray(valid, dtype=int)
