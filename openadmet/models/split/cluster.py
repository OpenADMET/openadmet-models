"""Cluster-based data splitting implementations."""

import logging
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans
from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
import datamol as dm
import numpy as np
import pandas as pd
from openadmet.models.split.split_base import SplitterBase, splitters
from useful_rdkit_utils import (
    get_butina_clusters,
    get_bemis_murcko_clusters,
    get_scaffold,
    smi2numpy_fp,
)


@splitters.register("ClusterSplitter")
class ClusterSplitter(SplitterBase):
    """Splits the data based on the KMeans clustering of the molecules."""

    method: str = "butina"
    k_clusters: int = 10
    kmeans_fp_type: str = "morgan"
    butina_cutoff: float = 0.65

    @field_validator("method", mode="before")
    @classmethod
    def validate_method(cls, value):
        """Validate that the method is one of the allowed options."""
        if value not in {"butina", "kmeans", "bemis-murcko"}:
            raise ValueError(
                f"Invalid method: {value}. Must be one of 'butina', 'kmeans', or 'bemis-murcko'."
            )
        return value

    @model_validator(mode="after")
    def check_sizes(self):
        """Validate the sizes of the splits."""
        # Check that sizes sum to 1
        if self.test_size + self.val_size + self.train_size != 1.0:
            raise ValueError("Test and train sizes must sum to 1.0")

        # Check that val_size and test_size are not both 0
        if self.val_size + self.test_size == 0.0:
            raise ValueError("Either val_size or test_size must be greater than 0")

        return self

    def split(self, X, y, num_iters=1000):
        """
        Split the data into train, validation, and test sets.

        Parameters
        ----------
        X : Iterable[str]
            List or iterable of SMILES strings to split.
        y : Iterable[float] or pd.Series
            List or iterable of target values corresponding to the SMILES strings.
        num_iters : int, optional
            Number of Monte Carlo trials to minimize the deviation from target ratios. Default is 1000.

        Returns
        -------
        tuple
            Tuple containing:
            - X_train: Training set SMILES strings.
            - X_val: Validation set SMILES strings (or None if val_size=0).
            - X_test: Test set SMILES strings (or None if test_size=0).
            - y_train: Training set target values.
            - y_val: Validation set target values (or None if val_size=0).
            - y_test: Test set target values (or None if test_size=0).

        """
        # Get clusters based on the selected method
        if self.method == "butina":
            clusters = get_butina_clusters(X, cutoff=self.butina_cutoff)
        elif self.method == "bemis-murcko":
            clusters = get_bemis_murcko_clusters(X)
        elif self.method == "kmeans":
            logging.warning(
                "KMeans clustering is NOT DETERMINISTIC with random seed across platforms."
            )
            km = KMeans(
                n_clusters=self.k_clusters,
                n_init=1,
                random_state=self.random_state,
                algorithm="lloyd",
            )
            vec_featurizer = FPVecTransformer(self.kmeans_fp_type)
            transformer = MoleculeTransformer(
                vec_featurizer,
                parallel_kwargs={"progress": False},
            )
            with dm.without_rdkit_log():
                feat, _ = transformer(X, ignore_errors=True)
            fp_list = list(np.squeeze(feat))
            clusters = km.fit_predict(np.stack(fp_list, dtype=np.float64))

        # Group X into subarrays based on cluster assignments
        unique_clusters = np.unique(clusters)
        subarrays_X = [X[clusters == cluster] for cluster in unique_clusters]
        subarrays_y = [y[clusters == cluster] for cluster in unique_clusters]
        n_subarrays = len(subarrays_X)

        # Set subarray data
        lengths = np.array([len(arr) for arr in subarrays_X])
        total_elements = lengths.sum()
        indices = np.arange(n_subarrays)
        ratios = [self.train_size, self.val_size, self.test_size]
        ratios = np.cumsum(ratios)[:2]
        target_counts = (ratios * total_elements).astype(int)

        best_split = None
        min_error = float("inf")
        rng = np.random.default_rng(self.random_state)

        # Search for best set of clusters to split with specified sizes
        for _ in range(num_iters):
            shuffled_indices = rng.permutation(indices)

            # Calculate cumulative sum of lengths in this shuffled order
            shuffled_lengths = lengths[shuffled_indices]
            counts = np.cumsum(shuffled_lengths)

            # Searchsorted finds the first index where cum_counts >= target
            split_1 = np.searchsorted(counts, target_counts[0])
            split_2 = np.searchsorted(counts, target_counts[1])

            # Look at how far the actual cut points are from ideal targets
            error = abs(counts[split_1] - target_counts[0]) + abs(
                counts[split_2] - target_counts[1]
            )

            if error < min_error:
                min_error = error
                best_split = (shuffled_indices, split_1, split_2)

        best_indices, s1, s2 = best_split

        train_idxs = best_indices[: s1 + 1]
        val_idxs = best_indices[s1 + 1 : s2 + 1]
        test_idxs = best_indices[s2 + 1 :]

        # Retrieve train, val, and test sets for X and y separately
        X_train, X_val, X_test = retrieve_data_by_idx(
            subarrays_X, [train_idxs, val_idxs, test_idxs]
        )
        y_train, y_val, y_test = retrieve_data_by_idx(
            subarrays_y, [train_idxs, val_idxs, test_idxs]
        )

        if (
            self.train_size != len(X_train)
            or self.val_size != len(X_val) / total_elements
            or self.test_size != len(X_test) / total_elements
        ):
            logging.warning(
                f"Train/val/test sizes DO NOT match input requests due to cluster sizes: Train: {self.train_size / total_elements}, Val: {self.val_size / total_elements}, Test: {self.test_size / total_elements}"
            )

        # Return train, val and test sets
        return X_train, X_val, X_test, y_train, y_val, y_test, clusters


def retrieve_data_by_idx(subarrays, all_inds):
    """Retrieve data based on indices."""
    to_return = []
    for idxs in all_inds:
        if len(idxs) == 0:
            to_return.append(None)
        else:
            items = [subarrays[i] for i in idxs]

            if isinstance(items[0], pd.Series):
                to_return.append(pd.concat(items))
            elif isinstance(items[0], pd.DataFrame):
                to_return.append(pd.concat(items, axis=0))
            else:
                to_return.append(np.concatenate(items))

    return to_return
