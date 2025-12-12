"""Cluster-based data splitting implementations."""

import logging
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
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

    def split(self, X, y):
        """
        Split the data into train, validation, and test sets.

        Parameters
        ----------
        X : Iterable[str]
            List or iterable of SMILES strings to split.
        y : Iterable[float] or pd.Series
            List or iterable of target values corresponding to the SMILES strings.

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
            km = KMeans(
                n_clusters=self.k_clusters,
                n_init=1,
                random_state=self.random_state,
                algorithm="lloyd",
            )
            fp_list = [smi2numpy_fp(x).astype(np.float64) for x in X]
            with threadpool_limits(limits=1):
                clusters = km.fit_predict(np.stack(fp_list))

        # Calculate and log the frequency of each cluster
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        cluster_frequencies = dict(zip(unique_clusters, counts))
        logging.warning(f"Cluster frequencies: {cluster_frequencies}")
        freqs = {0: 7, 1: 31, 2: 42, 3: 14, 4: 61, 5: 14, 6: 3, 7: 16, 8: 4, 9: 64, 10: 16, 11: 209, 12: 21, 13: 13, 14: 22, 15: 20, 16: 4, 17: 20, 18: 11, 19: 34, 20: 10, 21: 17, 22: 26, 23: 11, 24: 14, 25: 8, 26: 12, 27: 16, 28: 7, 29: 14, 30: 29, 31: 47, 32: 15, 33: 17, 34: 93, 35: 11, 36: 20, 37: 20, 38: 9, 39: 19, 40: 46, 41: 10, 42: 23, 43: 11, 44: 28, 45: 23, 46: 21, 47: 18, 48: 12, 49: 7, 50: 37, 51: 21, 52: 76, 53: 32, 54: 11, 55: 13, 56: 11, 57: 7, 58: 13, 59: 8, 60: 5, 61: 17, 62: 15, 63: 14, 64: 18, 65: 4, 66: 7, 67: 37, 68: 25, 69: 13, 70: 7, 71: 6, 72: 19, 73: 5, 74: 7, 75: 24, 76: 6, 77: 58, 78: 14, 79: 10, 80: 7, 81: 15, 82: 18, 83: 2, 84: 20, 85: 18, 86: 8, 87: 20, 88: 8, 89: 5, 90: 5, 91: 9, 92: 15, 93: 8, 94: 18, 95: 8, 96: 15, 97: 14, 98: 4, 99: 3}
        are_equal = freqs == cluster_frequencies
        logging.warning(f"Are freqs and cluster_frequencies equal? {are_equal}")

        if self.test_size == 0 and self.val_size == 0:
            X_train, y_train = X, y
            return X, None, None, y, None, None, clusters

        if self.test_size == 0:
            logging.warning(
                "val_size " + str(int(1 / self.val_size)) + " " + str(self.val_size)
            )
            # Split into train and val
            gss = GroupShuffleSplit(
                n_splits=int(1 / self.val_size), random_state=self.random_state
            )
            for train_idx, val_idx in gss.split(X, y, groups=clusters):
                X_train, X_val = np.array(X)[train_idx], np.array(X)[val_idx]
                y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]
                break
            return X_train, X_val, None, y_train, y_val, None, clusters

        logging.warning(
            "test_size " + str(int(1 / self.test_size)) + " " + str(self.test_size)
        )
        gss = GroupShuffleSplit(
            n_splits=int(1 / self.test_size), random_state=self.random_state
        )
        for train_val_idx, test_idx in gss.split(X, y, groups=clusters):
            X_train_val, X_test = np.array(X)[train_val_idx], np.array(X)[test_idx]
            y_train_val, y_test = np.array(y)[train_val_idx], np.array(y)[test_idx]
            break

        if self.val_size == 0:
            return X_train_val, None, X_test, y_train_val, None, y_test, clusters

        logging.warning(
            "val_size " + str(int(1 / self.val_size)) + " " + str(self.val_size)
        )
        gss = GroupShuffleSplit(
            n_splits=int(1 / self.val_size), random_state=self.random_state
        )
        for train_idx, val_idx in gss.split(
            X_train_val, y_train_val, groups=np.array(clusters)[train_val_idx]
        ):
            X_train, X_val = (
                np.array(X_train_val)[train_idx],
                np.array(X_train_val)[val_idx],
            )
            y_train, y_val = (
                np.array(y_train_val)[train_idx],
                np.array(y_train_val)[val_idx],
            )
            break

        # Return train, val and test sets
        return X_train, X_val, X_test, y_train, y_val, y_test, clusters
