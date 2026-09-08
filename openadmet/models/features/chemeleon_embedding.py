"""CheMeleon embedding featurizer."""

from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch
from pydantic import field_validator

from openadmet.models.architecture.chemprop import (
    _CHEMELEON_MP_HPARAMS,
    ChemPropModel,
    _resolve_device,
)
from openadmet.models.features.feature_base import FeaturizerBase, featurizers


# Foundation checkpoint, patched by tests to the weightless architecture
_FOUNDATION_NAME = "chemeleon"

# Zero-row width, so an empty input skips the checkpoint download
_FOUNDATION_EMBEDDING_DIM: int = _CHEMELEON_MP_HPARAMS["d_h"]


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
        Device to use for inference. "auto" resolves like Lightning's auto
        accelerator: TPU, MPS, or CUDA when available, otherwise CPU.
    batch_size : int
        Number of molecules per forward pass.

    Attributes
    ----------
    model : ChemPropModel
        Encoder built on first access and reused across calls, so repeated
        featurization does not reload the checkpoint.

    """

    type: ClassVar[str] = "CheMeleonEmbeddingFeaturizer"

    accelerator: str = "auto"
    batch_size: int = 256

    _model: ChemPropModel | None = None

    @field_validator("accelerator")
    @classmethod
    def validate_accelerator(cls, value: str) -> str:
        """
        Validate that the accelerator resolves to a torch-recognized device.

        Reproduces the torch.device() check eagerly here so a bad accelerator
        fails at construction time rather than inside predict_embedding.

        Parameters
        ----------
        value : str
            Accelerator value to validate.

        Returns
        -------
        str
            The validated accelerator value.

        """
        try:
            torch.device(_resolve_device(value))
        except RuntimeError as e:
            raise ValueError(f"Invalid accelerator {value!r}: {e}") from e
        return value

    @property
    def model(self) -> ChemPropModel:
        """Return the CheMeleon encoder, building it on first access."""
        if self._model is None:
            # Cache only after build() succeeds, so a failure is not memoized
            model = ChemPropModel(from_foundation=_FOUNDATION_NAME)
            model.build()
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

        # Shaped from the constant, so an empty input never builds the model
        if not smiles_list:
            return (
                np.empty((0, _FOUNDATION_EMBEDDING_DIM), dtype=np.float32),
                np.empty(0, dtype=int),
            )

        # The featurizer owns the device choice, so pass it rather than default
        embeddings = self.model.predict_embedding(
            smiles_list, batch_size=self.batch_size, accelerator=self.accelerator
        )

        # Every input row is featurized, so rows map 1:1 to input positions
        return embeddings, np.arange(len(smiles_list), dtype=int)
