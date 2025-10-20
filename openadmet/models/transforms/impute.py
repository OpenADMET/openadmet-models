"""Imputation transforms for handling missing data."""

from openadmet.models.transforms.transform_base import TransformBase, transforms
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import numpy as np
from pydantic import field_validator
from typing import Optional, Literal


@transforms.register("ImputeTransform")
class ImputeTransform(TransformBase):
    """
    Impute missing values in the dataset using a specified strategy.

    Attributes
    ----------
    strategy : str
        The imputation strategy to use. Options are 'mean', 'median', 'most_frequent', or 'constant'.
    imputer : str
        The type of imputer to use. Options are 'simple' for SimpleImputer or
        'iterative' for IterativeImputer.
    random_state : Optional[int]
        Random state for reproducibility when using IterativeImputer.

    """

    strategy: str = (
        "mean"  # Default strategy is to replace missing values with the mean
    )
    imputer: Literal["simple", "iterative"] = "simple"  # Can be 'simple' or 'iterative'
    random_state: Optional[int] = None  # Optional random state for reproducibility
    imputer_instance: Optional[object] = None  # Placeholder for the imputer instance

    @field_validator("strategy")
    def validate_strategy(cls, value):
        """Validate the strategy parameter."""
        if value not in ["mean", "median", "most_frequent", "constant"]:
            raise ValueError(
                "Strategy must be one of 'mean', 'median', 'most_frequent', or 'constant'"
            )
        return value

    @field_validator("imputer")
    def validate_imputer(cls, value):
        """Validate the imputer type."""
        if value not in ["simple", "iterative"]:
            raise ValueError("Imputer must be either 'simple' or 'iterative'")
        return value

    def fit(self, X: np.ndarray, *args, **kwargs):
        """
        Fit the imputer to the data X.

        Parameters
        ----------
        X: np.ndarray
            Input data with potential missing values
        *args
            Additional positional arguments (not used).
        **kwargs
            Additional keyword arguments (not used).

        Returns
        -------
        self
            Fitted transform instance

        """
        if self.imputer == "simple":
            self.imputer_instance = SimpleImputer(strategy=self.strategy)
        elif self.imputer == "iterative":
            self.imputer_instance = IterativeImputer(random_state=self.random_state)
        else:
            raise ValueError("Invalid imputer type specified.")

        self.imputer_instance.fit(X)

    def transform(self, X: np.ndarray, *args, **kwargs):
        """
        Impute missing values in the input data X.

        Parameters
        ----------
        X: np.ndarray
            Input data with potential missing values
        *args
            Additional positional arguments (not used).
        **kwargs
            Additional keyword arguments (not used).

        Returns
        -------
        np.ndarray
            Transformed data with missing values imputed

        """
        if self.imputer_instance is None:
            raise RuntimeError("The imputer has not been fitted yet. Call 'fit' first.")
        return self.imputer_instance.transform(X)
