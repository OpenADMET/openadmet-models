"""Base classes and utilities for molecular featurizers."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from class_registry import ClassRegistry, RegistryKeyError
from molfeat.trans import MoleculeTransformer
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

featurizers = ClassRegistry(unique=True)


def get_featurizer_class(feat_type):
    """Retrieve a featurizer class from the registry by type."""
    try:
        feat_class = featurizers.get_class(feat_type)
    except RegistryKeyError:
        raise ValueError(f"Feature type {feat_type} not found in feature catalouge")
    return feat_class


class FeaturizerBase(BaseModel, ABC):
    """
    Base class for featurizers, allowing for arbitrary featurization of molecules.

    This class defines the interface for all featurizers. Subclasses should implement
    the `featurize` method to convert a list of SMILES strings into features suitable
    for machine learning models.

    """

    @abstractmethod
    def featurize(self, smiles: Iterable[str], *args, **kwargs):
        """
        Featurize a list of SMILES strings.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings to featurize.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Any
            Features in an appropriate format for the model (e.g., numpy arrays, dataloaders, etc.)
            and optional processing info.

        """
        pass


class DeepLearningFeaturizer(FeaturizerBase):
    """
    Base class for deep learning featurizers.

    This class extends FeaturizerBase and standardizes the output for deep learning workflows.
    Subclasses should implement the `featurize` method to return a DataLoader, indices,
    a StandardScaler, and a PyTorch Dataset.

    """

    @abstractmethod
    def featurize(
        self, smiles: Iterable[str], y: Iterable[float] = None
    ) -> tuple[DataLoader, np.ndarray, StandardScaler, Dataset]:
        """
        Featurize a list of SMILES strings for deep learning models.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings to featurize.
        y : Iterable[float], optional
            Target values corresponding to the SMILES strings.

        Returns
        -------
        tuple
            Tuple containing:
            - DataLoader: PyTorch DataLoader for the dataset.
            - np.ndarray: Array of indices corresponding to the original input.
            - StandardScaler: Scaler used for any scaling during featurization.
            - Dataset: PyTorch Dataset containing the features and targets.

        """
        pass


class MolfeatFeaturizer(FeaturizerBase):
    """
    Featurizer using molfeat.

    This class provides a base for featurizers that use the molfeat library.
    It manages a MoleculeTransformer instance for feature extraction.

    Attributes
    ----------
    _transformer : MoleculeTransformer
        The underlying molfeat transformer used for featurization.

    """

    _transformer: MoleculeTransformer = None

    def __init__(self, *args, **kwargs):
        """
        Initialize the MolfeatFeaturizer.

        Parameters
        ----------
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        """
        super().__init__(*args, **kwargs)
        self._prepare()

    @abstractmethod
    def _prepare(self):
        """
        Prepare the featurizer.

        This method should be implemented by subclasses to initialize or configure
        the underlying molfeat transformer.
        """
        pass

    @property
    def transformer(self):
        """Return the transformer, for use in SkLearn pipelines etc."""
        return self._transformer
