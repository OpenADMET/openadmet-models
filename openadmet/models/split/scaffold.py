from sklearn.model_selection import train_test_split
from splito import MaxDissimilaritySplit, PerimeterSplit, ScaffoldSplit
import numpy as np
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
                X[train_idx],
                X[val_idx],
                None,
                y[train_idx],
                y[val_idx],
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
                X[train_val_idx],
                None,
                X[test_idx],
                y[train_val_idx],
                None,
                y[test_idx],
            )

        # Split train+val into train and val sets
        X_train, X_val, y_train, y_val = train_test_split(
            X[train_val_idx],
            y[train_val_idx],
            train_size=None,
            test_size=int(self.val_size * X.shape[0]),
            random_state=self.random_state,
        )

        # Return train, val, and test sets
        return (
            X_train,
            X_val,
            X[test_idx],
            y_train,
            y_val,
            y[test_idx],
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
                X[train_idx],
                X[val_idx],
                None,
                y[train_idx],
                y[val_idx],
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
                X[train_val_idx],
                None,
                X[test_idx],
                y[train_val_idx],
                None,
                y[test_idx],
            )

        # Split train+val into train and val sets using sklearn
        X_train, X_val, y_train, y_val = train_test_split(
            X[train_val_idx],
            y[train_val_idx],
            train_size=None,
            test_size=int(self.val_size * X.shape[0]),
            random_state=self.random_state,
        )

        # Return train, val, and test sets
        return (
            X_train,
            X_val,
            X[test_idx],
            y_train,
            y_val,
            y[test_idx],
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
                X[train_idx],
                X[val_idx],
                None,
                y[train_idx],
                y[val_idx],
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
                X[train_val_idx],
                None,
                X[test_idx],
                y[train_val_idx],
                None,
                y[test_idx],
            )

        # Split train+val into train and val sets using sklearn
        X_train, X_val, y_train, y_val = train_test_split(
            X[train_val_idx],
            y[train_val_idx],
            train_size=None,
            test_size=int(self.val_size * X.shape[0]),
            random_state=self.random_state,
        )

        # Return train, val and test sets
        return (
            X_train,
            X_val,
            X[test_idx],
            y_train,
            y_val,
            y[test_idx],
        )
