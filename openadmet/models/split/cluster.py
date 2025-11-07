"""Cluster-based data splitting implementations."""

from sklearn.model_selection import train_test_split
from splito import KMeansSplit
import numpy as np
import pandas as pd
from openadmet.models.split.split_base import SplitterBase, splitters
from openadmet.models.split.scaffold import safe_index
from useful_rdkit_utils import (
    GroupKFoldShuffle,
    get_butina_clusters,
    get_bemis_murcko_clusters,
)


@splitters.register("KMeansSplitter")
class KMeansSplitter(SplitterBase):
    """Splits the data based on the KMeans clustering of the molecules."""

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
        # No test set requested
        if self.test_size == 0:
            # Split into train and val
            splitter = KMeansSplit(
                smiles=X,
                n_jobs=-1,
                train_size=None,
                test_size=int(self.val_size * X.shape[0]),
                random_state=self.random_state,
            )
            train_idx, val_idx = next(splitter.split(X=X))

            return (
                safe_index(X, train_idx),
                safe_index(X, val_idx),
                None,
                safe_index(y, train_idx),
                safe_index(y, val_idx),
                None,
            )

        # Split into train+val and test
        splitter = KMeansSplit(
            n_jobs=-1,
            train_size=None,
            test_size=int(self.test_size * X.shape[0]),
            random_state=self.random_state,
        )
        train_val_idx, test_idx = next(splitter.split(X=X))

        # No validation set requested, return train(+val) and test sets
        if self.val_size == 0:
            return (
                safe_index(X, train_val_idx),
                None,
                safe_index(X, test_idx),
                safe_index(y, train_val_idx),
                None,
                safe_index(y, test_idx),
            )

        # Split train+val into train and val sets using sklearn
        X_train, X_val, y_train, y_val = train_test_split(
            safe_index(X, train_val_idx),
            safe_index(y, train_val_idx),
            train_size=None,
            test_size=int(self.val_size * X.shape[0]),
            random_state=self.random_state,
        )

        # Return train, val, and test sets
        return (
            X_train,
            X_val,
            safe_index(X, test_idx),
            y_train,
            y_val,
            safe_index(y, test_idx),
        )


class ButinaSplitter(SplitterBase):
    """Splits the data based on the Butina clustering of the molecules."""

    def split(self, X, y):
        """Split the data into train, validation, and test sets."""
        clusters = get_butina_clusters(X)


class BemisMurckoSplitter(SplitterBase):
    """Splits the data based on the Bemis-Murcko clustering of the molecules."""

    def split(self, X, y):
        """Split the data into train, validation, and test sets."""
        clusters = get_bemis_murcko_clusters(X)
