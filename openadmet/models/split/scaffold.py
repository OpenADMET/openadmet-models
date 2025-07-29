from sklearn.model_selection import train_test_split
from splito import MaxDissimilaritySplit, PerimeterSplit, ScaffoldSplit
import numpy as np
import pandas as pd
from openadmet.models.split.split_base import SplitterBase, splitters

@splitters.register("ScaffoldSplitter")
class ScaffoldSplitter(SplitterBase):
    """
    Splits the data based on the scaffold of the molecules
    """

    def split(self, X, y):
        """
        Split the data into train, validation, and test sets
        """

        # No test set requested
        if self.test_size == 0:
            # Split into train and val
            splitter = ScaffoldSplit(
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
        splitter = ScaffoldSplit(
            smiles=X,
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

        # Split train+val into train and val sets
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


@splitters.register("PerimeterSplitter")
class PerimeterSplitter(SplitterBase):
    """
    Splits the data based on the perimeter of the molecules
    """

    def split(self, X, y):
        """
        Split the data into train, validation, and test sets
        """
        # No test set requested
        if self.test_size == 0:
            # Split into train and val
            splitter = PerimeterSplit(
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
        splitter = PerimeterSplit(
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


@splitters.register("MaxDissimilaritySplitter")
class MaxDissimilaritySplitter(SplitterBase):
    """
    Splits the data based on maximum dissimilarity
    """

    def split(self, X, y):
        """
        Split the data into train, validation, and test sets
        """
        # No test set requested
        if self.test_size == 0:
            # Split into train and val
            splitter = MaxDissimilaritySplit(
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
        splitter = MaxDissimilaritySplit(
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

        # Return train, val and test sets
        return (
            X_train,
            X_val,
            safe_index(X, test_idx),
            y_train,
            y_val,
            safe_index(y, test_idx),
        )

def safe_index(data, idx):
    """A helper function for correct indexing depending on whether X and y are numpy arrays or pandas series/dataframes.

    Parameters
    ----------
    data : nd.array, list, pd.Series, or pd.DataFrame
        X or y data
    idx : list
        list of integers (positional indices)

    Returns
    -------
    nd.array or pd.Series
        indexed data
    """
    if isinstance(data, (np.ndarray, list)):
        return data[idx]
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.iloc[idx]
    else:
        raise TypeError(f"Unsupported data type for indexing: {type(data)}")
