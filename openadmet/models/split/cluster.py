"""Cluster-based data splitting implementations."""

from pydantic import BaseModel, field_validator, model_validator
from typing import Literal
from sklearn.model_selection import GroupShuffleSplit
from splito import KMeansSplit
import numpy as np
import pandas as pd
from openadmet.models.split.split_base import SplitterBase, splitters
from useful_rdkit_utils import (
    get_butina_clusters,
    get_bemis_murcko_clusters,
    get_scaffold,
    smi2numpy_fp,
)
from sklearn.cluster import KMeans


@splitters.register("ClusterSplitter")
class ClusterSplitter(SplitterBase):
    """Splits the data based on the KMeans clustering of the molecules."""

    method: str = "butina"
    k_clusters: int = 10

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
            clusters = get_butina_clusters(X)
        elif self.method == "bemis-murcko":
            clusters = get_bemis_murcko_clusters(X)
        elif self.method == "kmeans":
            km = KMeans(
                n_clusters=self.k_clusters,
                n_init="auto",
                random_state=self.random_state,
            )
            fp_list = [smi2numpy_fp(x) for x in X]
            clusters = km.fit_predict(np.stack(fp_list))

        if self.test_size == 0 and self.val_size == 0:
            X_train, y_train = X, y
            return X, None, None, y, None, None, clusters

        if self.test_size == 0:
            # Split into train and val
            gss = GroupShuffleSplit(
                n_splits=int(1 / self.val_size), random_state=self.random_state
            )
            for train_idx, val_idx in gss.split(X, y, groups=clusters):
                X_train, X_val = np.array(X)[train_idx], np.array(X)[val_idx]
                y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]
                break
            return X_train, X_val, None, y_train, y_val, None, clusters

        gss = GroupShuffleSplit(
            n_splits=int(1 / self.test_size), random_state=self.random_state
        )
        for train_val_idx, test_idx in gss.split(X, y, groups=clusters):
            X_train_val, X_test = np.array(X)[train_val_idx], np.array(X)[test_idx]
            y_train_val, y_test = np.array(y)[train_val_idx], np.array(y)[test_idx]
            break

        if self.val_size == 0:
            return X_train_val, None, X_test, y_train_val, None, y_test, clusters

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
