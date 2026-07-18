"""ChemProp featurizer implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Union

import numpy as np
import pandas as pd

from openadmet.models.features.feature_base import DeepLearningFeaturizer, featurizers


# we vendor this from chemprop so that we can pass custom samplers
# taken directly from https://github.com/chemprop/chemprop/blob/main/chemprop/data/dataloader.py
def _vendor_build_dataloader(
    dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    sampler: Any = None,
    seed: int | None = None,
    shuffle: bool = True,
    drop_last_on_singleton: bool = True,
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
    drop_last_on_singleton : bool, default=True
        Whether to drop a size-1 final batch (when ``len(dataset) % batch_size == 1``) to avoid
        batch-norm errors. Set False for evaluation and inference loaders so every row is returned.
    **kwargs
        Additional keyword arguments passed to the DataLoader.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader for the given MoleculeDataset, ReactionDataset, or MulticomponentDataset.

    """
    from chemprop.data import MulticomponentDataset
    from chemprop.data.collate import collate_batch, collate_multicomponent
    from chemprop.data.samplers import ClassBalanceSampler, SeededSampler
    from torch.utils.data import DataLoader

    if sampler is None:
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

    # Drop a size-1 final batch only when requested (training), to avoid batch-norm errors
    drop_last = drop_last_on_singleton and len(dataset) % batch_size == 1

    return DataLoader(
        dataset,
        batch_size,
        sampler is None and shuffle,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        # keep workers alive across epochs to avoid re-spawning pipes each epoch,
        # which exhausts file descriptors on long runs
        persistent_workers=num_workers > 0,
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
    left_censor_threshold : float or None, optional
        Lower detection limit (raw target units) below which a measurement is treated as
        left-censored rather than exact, by default None (no censoring). When set, training
        targets below the threshold are clamped up to it and flagged with chemprop's
        ``lt_mask``, so a paired :class:`CensoredRegressionLoss` scores them as "below the
        bound" instead of as exact regression targets that anchor the fit. Censoring is
        applied only to the training loader (``train=True``); validation and inference keep
        exact, unclamped targets so the monitored loss and the held-out evaluation read
        against true values.

    """

    normalize_targets: bool = True
    n_jobs: int = 4
    batch_size: int = 128
    shuffle: bool = False
    left_censor_threshold: float | None = None

    def _prepare(self):
        """Prepare the featurizer."""

    def featurize(
        self, smiles: Iterable[str], y: Iterable[Any] = None, train: bool = False
    ) -> tuple[
        DataLoader,
        np.ndarray,
        StandardScaler,
        MoleculeDataset | ReactionDataset | MulticomponentDataset,
    ]:
        """
        Featurize a list of SMILES strings.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings to featurize.
        y : Iterable[Any], optional
            Target values corresponding to the SMILES strings.
        train : bool, optional
            Whether this loader feeds model training, by default False. Shuffling and the
            batch-norm ``drop_last`` guard apply only when True; otherwise the loader
            preserves input order and returns every row.

        Returns
        -------
        tuple
            Tuple containing:
            - DataLoader: PyTorch DataLoader for the dataset.
            - np.ndarray: Array of indices corresponding to the original input.
            - StandardScaler: Scaler used for any scaling during featurization.
            - Union[MoleculeDataset, ReactionDataset, MulticomponentDataset]: PyTorch Dataset containing the features and targets.

        """
        from chemprop.data import MoleculeDatapoint, MoleculeDataset

        if y is not None:
            # if a pandas dataframe or series
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.to_numpy()
            y = y.reshape(-1, 1) if y.ndim == 1 else y

            # below the detection limit a measurement is only "less than the bound": clamp the
            # target up to the bound and flag it so a censored loss penalizes only predictions
            # above it. Censoring applies only to the training loader; validation/inference
            # keep exact, unclamped targets.
            lt_masks = None
            y_used = y
            if self.left_censor_threshold is not None and train:
                lt_masks = np.asarray(y) < self.left_censor_threshold
                y_used = np.where(lt_masks, self.left_censor_threshold, y)

            lt_rows = lt_masks if lt_masks is not None else [None] * len(smiles)
            datapoints = []
            for smi, y_, lt_ in zip(smiles, y_used, lt_rows):
                extra = {"lt_mask": lt_} if lt_ is not None else {}
                datapoints.append(MoleculeDatapoint.from_smi(smi, y_, **extra))
            dataset = MoleculeDataset(datapoints)
            if self.normalize_targets:
                if lt_masks is not None:
                    # fit the target scaler on the unclamped (true) values, not the clamped
                    # ones, so normalization is identical to an uncensored run; only the loss
                    # treats sub-limit rows as censored. Clamping before fitting would pile
                    # mass at the bound, shift the mean up and shrink the std, and that
                    # re-normalization alone would lift predictions and confound the censoring
                    # effect with a trivial target rescaling.
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler().fit(np.asarray(y))
                    dataset.normalize_targets(scaler)
                else:
                    scaler = dataset.normalize_targets()
            else:
                scaler = None
        else:
            dataset = MoleculeDataset(
                [MoleculeDatapoint.from_smi(smi) for smi in smiles]
            )
            scaler = None

        # Shuffle and the size-1 drop_last guard are training-only; evaluation and
        # inference loaders preserve input order and length for correct y_true/y_pred pairing.
        # Passing the seed makes the training shuffle reproducible via SeededSampler
        dataloader = self.dataset_to_dataloader(
            dataset,
            num_workers=self.n_jobs,
            shuffle=self.shuffle and train,
            batch_size=self.batch_size,
            drop_last_on_singleton=train,
            seed=self.random_seed if train else None,
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

    def make_new(self) -> ChemPropFeaturizer:
        """Copy parameters to a new ChemPropFeaturizer instance."""
        return self.__class__(**self.dict())
