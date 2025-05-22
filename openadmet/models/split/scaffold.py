from splito import MaxDissimilaritySplit, PerimeterSplit, ScaffoldSplit

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
        # First split into train and val+test
        splitter = ScaffoldSplit(
            smiles=X,
            n_jobs=-1,
            train_size=None,
            test_size=int((self.test_size + self.val_size) * X.shape[0]),
            random_state=self.random_state,
        )
        train_idx, val_test_idx = next(splitter.split(X=X))

        # If no validation set is requested, return train and test sets
        if self.val_size == 0:
            return (
                X[train_idx],
                None,
                X[val_test_idx],
                y[train_idx],
                None,
                y[val_test_idx],
            )

        # If no test set is requested, return train and validation sets
        elif self.test_size == 0:
            return (
                X[train_idx],
                X[val_test_idx],
                None,
                y[train_idx],
                y[val_test_idx],
                None,
            )

        # If both test and validation sets are requested, split the remaining data
        else:
            val_test_splitter = ScaffoldSplit(
                smiles=X[val_test_idx],
                n_jobs=-1,
                train_size=None,
                test_size=int(self.test_size * X.shape[0]),
                random_state=self.random_state,
            )
            val_idx, test_idx = next(val_test_splitter.split(X=X[val_test_idx]))

            return (
                X[train_idx],
                X[val_idx],
                X[test_idx],
                y[train_idx],
                y[val_idx],
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

        # First split into train and val+test
        splitter = PerimeterSplit(
            n_jobs=-1,
            train_size=None,
            test_size=int((self.test_size + self.val_size) * X.shape[0]),
            random_state=self.random_state,
        )
        train_idx, val_test_idx = next(splitter.split(X=X))

        # If no validation set is requested, return train and test sets
        if self.val_size == 0:
            return (
                X[train_idx],
                None,
                X[val_test_idx],
                y[train_idx],
                None,
                y[val_test_idx],
            )

        # If no test set is requested, return train and validation sets
        elif self.test_size == 0:
            return (
                X[train_idx],
                X[val_test_idx],
                None,
                y[train_idx],
                y[val_test_idx],
                None,
            )

        # If both test and validation sets are requested, split the remaining data
        else:
            val_test_splitter = PerimeterSplit(
                n_jobs=-1,
                train_size=None,
                test_size=int(self.test_size * X.shape[0]),
                random_state=self.random_state,
            )
            val_idx, test_idx = next(val_test_splitter.split(X=X[val_test_idx]))

            return (
                X[train_idx],
                X[val_idx],
                X[test_idx],
                y[train_idx],
                y[val_idx],
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

        # First split into train and val+test
        splitter = MaxDissimilaritySplit(
            n_jobs=-1,
            train_size=None,
            test_size=int((self.test_size + self.val_size) * X.shape[0]),
            random_state=self.random_state,
        )
        train_idx, val_test_idx = next(splitter.split(X=X))

        # If no validation set is requested, return train and test sets
        if self.val_size == 0:
            return (
                X[train_idx],
                None,
                X[val_test_idx],
                y[train_idx],
                None,
                y[val_test_idx],
            )

        # If no test set is requested, return train and validation sets
        elif self.test_size == 0:
            return (
                X[train_idx],
                X[val_test_idx],
                None,
                y[train_idx],
                y[val_test_idx],
                None,
            )

        # If both test and validation sets are requested, split the remaining data
        else:
            val_test_splitter = MaxDissimilaritySplit(
                n_jobs=-1,
                train_size=None,
                test_size=int(self.test_size * X.shape[0]),
                random_state=self.random_state,
            )
            val_idx, test_idx = next(val_test_splitter.split(X=X[val_test_idx]))

            return (
                X[train_idx],
                X[val_idx],
                X[test_idx],
                y[train_idx],
                y[val_idx],
                y[test_idx],
            )
