"""Base classes and utilities for molecular featurizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel

if TYPE_CHECKING:
    from molfeat.trans import MoleculeTransformer
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, Dataset

featurizers = ClassRegistry(unique=True)


def get_featurizer_class(feat_type):
    """
    Retrieve a featurizer class from the registry by type.

    Parameters
    ----------
    feat_type : str
        The registered key for the featurizer (e.g., ``"MolfeatFeaturizer"``).

    Returns
    -------
    type
        The featurizer class corresponding to the given type.

    Raises
    ------
    ValueError
        If ``feat_type`` is not found in the featurizer registry.

    """
    from openadmet.models._registry_loader import load_group

    load_group("featurizers")
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

    def feature_blocks(self, probe: Iterable[str]) -> list[tuple[str, int]]:
        """
        Return contiguous feature blocks as (key, n_features) pairs in column order.

        The default treats a featurizer as a single block keyed by its registry
        name and probes the width by featurizing the first probe entry that
        yields at least one row. Featurizers with known widths may override
        this to skip the probe.

        Parameters
        ----------
        probe : Iterable[str]
            Input rows (e.g., SMILES) to featurize; tried in order until one
            yields a non-empty feature matrix.

        Returns
        -------
        list of tuple
            Pairs of (block key, feature width) in the order columns appear.

        Raises
        ------
        ValueError
            If no probe entry yields a non-empty feature matrix.

        """
        key = getattr(self, "type", type(self).__name__)
        for candidate in probe:
            # A candidate that does not featurize (e.g. an unparseable SMILES) must be
            # skipped, not fatal: the probe is best-effort width discovery.
            try:
                feats, _ = self.featurize([candidate])
            except Exception:
                continue
            if feats is None or len(feats) == 0:
                continue
            arr = np.atleast_2d(np.asarray(feats))
            return [(key, arr.shape[1])]
        raise ValueError(
            f"Could not determine feature width for {key}: no probe entry featurized successfully."
        )


class DeepLearningFeaturizer(FeaturizerBase):
    """
    Base class for deep learning featurizers.

    This class extends FeaturizerBase and standardizes the output for deep learning workflows.
    Subclasses should implement the `featurize` method to return a DataLoader, indices,
    a StandardScaler, and a PyTorch Dataset.

    Attributes
    ----------
    random_seed : int or None
        Seed for the training DataLoader's shuffling, by default None. The legacy
        ``random_state`` name is accepted as a deprecated alias. Only applied when a
        training loader is built (``train=True``); evaluation and inference loaders
        are never shuffled and ignore it.

    """

    random_seed: int | None = None

    @abstractmethod
    def featurize(
        self, smiles: Iterable[str], y: Iterable[float] = None, train: bool = False
    ) -> tuple[DataLoader, np.ndarray, StandardScaler, Dataset]:
        """
        Featurize a list of SMILES strings for deep learning models.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings to featurize.
        y : Iterable[float], optional
            Target values corresponding to the SMILES strings.
        train : bool, optional
            Whether this loader feeds model training, by default False. Shuffling and
            the batch-norm ``drop_last`` guard apply only when True; validation, test,
            and inference loaders (train=False) always preserve input order and length.

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

    _transformer: Any = None

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
