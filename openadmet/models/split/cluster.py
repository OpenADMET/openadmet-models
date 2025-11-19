"""Cluster-based data splitting implementations."""

from pydantic import BaseModel, field_validator, model_validator
from typing import Literal
from sklearn.model_selection import train_test_split
from splito import KMeansSplit
import numpy as np
import pandas as pd
from openadmet.models.split.split_base import SplitterBase, splitters
from useful_rdkit_utils import (
    get_kmeans_clusters,
    get_butina_clusters,
    get_bemis_murcko_clusters,
)


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
            clusters = get_kmeans_clusters(X, n_clusters=self.k_clusters)

        # No test set requested
        if self.test_size == 0:
            # Split into train and val
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                train_size=None,
                test_size=int(self.val_size * X.shape[0]),
                random_state=self.random_state,
                stratify=clusters,
            )
            return X_train, X_val, None, y_train, y_val, None

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            train_size=None,
            test_size=int(self.test_size * X.shape[0]),
            random_state=self.random_state,
            stratify=clusters,
        )

        # No validation set requested, return train(+val) and test sets
        if self.val_size == 0:
            return X_train_val, None, X_test, y_train_val, None, y_test

        # Get new clusters based on the selected method
        if self.method == "butina":
            split_clusters = get_butina_clusters(X_train_val)
        elif self.method == "bemis-murcko":
            split_clusters = get_bemis_murcko_clusters(X_train_val)
        elif self.method == "kmeans":
            split_clusters = get_kmeans_clusters(
                X_train_val, n_clusters=self.k_clusters
            )

        # Split train+val into train and val sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            train_size=None,
            test_size=int(self.val_size * X.shape[0]),
            random_state=self.random_state,
            stratify=split_clusters,
        )

        # Return train, val and test sets
        return X_train, X_val, X_test, y_train, y_val, y_test
