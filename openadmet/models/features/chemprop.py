"""ChemProp featurizer implementation."""

from collections.abc import Iterable
from typing import Any, Union
import numpy as np
import pandas as pd
from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    MulticomponentDataset,
    ReactionDataset,
)
from chemprop.data.collate import collate_batch, collate_multicomponent
from chemprop.data.samplers import ClassBalanceSampler, SeededSampler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from openadmet.models.features.chemprop import (
    MoleculeDataset,
    MulticomponentDataset,
    ReactionDataset,
)
from openadmet.models.features.feature_base import DeepLearningFeaturizer, featurizers


# we vendor this from chemprop so that we can pass custom samplers
# taken directly from https://github.com/chemprop/chemprop/blob/main/chemprop/data/dataloader.py
def _vendor_build_dataloader(
    dataset: MoleculeDataset | ReactionDataset | MulticomponentDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    sampler: Any = None,
    seed: int | None = None,
    shuffle: bool = True,
    **kwargs,
):
    r"""
    Return a :obj:`~torch.utils.data.DataLoader` for :class:`MolGraphDataset`.

    Parameters
    ----------
    dataset : MoleculeDataset | ReactionDataset | MulticomponentDataset
        The dataset containing the molecules or reactions to load.
    batch_size : int, default=64
        the batch size to load.
    num_workers : int, default=0
        the number of workers used to build batches.
    class_balance : bool, default=False
        Whether to perform class balancing (i.e., use an equal number of positive and negative
        molecules). Class balance is only available for single task classification datasets. Set
        shuffle to True in order to get a random subset of the larger class.
    sampler : torch.utils.data.Sampler, optional
        Custom sampler to use for loading data (default is None). If this is specified, it
        overrides class_balance and shuffle.
    seed : int, optional
        Random seed for shuffling and class balancing (default is None).
    shuffle : bool, default=True
        Whether to shuffle the data at every epoch. If a sampler is specified, this is ignored
        (i.e., the sampler determines the shuffling). If class_balance is True, this is also ignored
        (i.e., class balancing determines the shuffling).
    **kwargs
        Additional keyword arguments passed to the DataLoader.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader for the given MoleculeDataset, ReactionDataset, or MulticomponentDataset.

    """
    if sampler is not None:
        if class_balance:
            sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
        elif shuffle and seed is not None:
            sampler = SeededSampler(len(dataset), seed)
        else:
            sampler = None

    if isinstance(dataset, MulticomponentDataset):
        collate_fn = collate_multicomponent
    else:
        collate_fn = collate_batch

    # Drop last batch of size 1 to avoid issues with batch normalization
    if len(dataset) % batch_size == 1:
        drop_last = True
    else:
        drop_last = False

    return DataLoader(
        dataset,
        batch_size,
        sampler is None and shuffle,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **kwargs,
    )


@featurizers.register("ChemPropFeaturizer")
class ChemPropFeaturizer(DeepLearningFeaturizer):
    """
    ChemPropFeaturizer featurizer for molecules, relies on chemprop.

    Parameters
    ----------
    normalize_targets : bool, optional
        Whether to normalize the targets using StandardScaler, by default True
    n_jobs : int, optional
        Number of parallel workers to use, by default 4
    batch_size : int, optional
        Batch size for the DataLoader, by default 128
    shuffle : bool, optional
        Whether to shuffle the data in the DataLoader, by default False

    """

    normalize_targets: bool = True
    n_jobs: int = 4
    batch_size: int = 128
    shuffle: bool = False

    def _prepare(self):
        """Prepare the featurizer."""

    def featurize(
        self, smiles: Iterable[str], y: Iterable[Any] = None
    ) -> tuple[
        DataLoader,
        np.ndarray,
        StandardScaler,
        Union[MoleculeDataset, ReactionDataset, MulticomponentDataset],
    ]:
        """
        Featurize a list of SMILES strings.

        #TODO: we likely want to separate the scaling from the featurization
        """
        if y is not None:
            # if a pandas dataframe or series
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.to_numpy()
            y = y.reshape(-1, 1) if y.ndim == 1 else y

            dataset = MoleculeDataset(
                [MoleculeDatapoint.from_smi(smi, y_) for smi, y_ in zip(smiles, y)]
            )
            if self.normalize_targets:
                scaler = dataset.normalize_targets()
            else:
                scaler = None
        else:
            dataset = MoleculeDataset(
                [MoleculeDatapoint.from_smi(smi) for smi in smiles]
            )
            scaler = None

        dataloader = self.dataset_to_dataloader(
            dataset,
            num_workers=self.n_jobs,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
        )

        # Need to also return an index of the original input for which the features were computed
        indices = np.arange(len(smiles))

        return dataloader, indices, scaler, dataset

    @staticmethod
    def dataset_to_dataloader(
        dataset: MoleculeDataset,
        batch_size: int = 128,
        shuffle: bool = False,
        sampler=None,
        **kwargs,
    ) -> DataLoader:
        """
        Convert a MoleculeDataset to a PyTorch DataLoader.

        Parameters
        ----------
        dataset : MoleculeDataset
            The dataset containing the molecules to load.
        batch_size : int, optional
            Number of samples per batch to load (default is 128).
        shuffle : bool, optional
            Whether to shuffle the data at every epoch (default is False).
        sampler : torch.utils.data.Sampler, optional
            Custom sampler to use for loading data (default is None).
        **kwargs
            Additional keyword arguments passed to the DataLoader.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader for the given MoleculeDataset.

        """
        return _vendor_build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs,
        )

    def make_new(self) -> "ChemPropFeaturizer":
        """Copy parameters to a new ChemPropFeaturizer instance."""
        return self.__class__(**self.dict())
