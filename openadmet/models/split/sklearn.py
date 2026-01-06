"""Sklearn-based data splitting implementations."""

from sklearn.model_selection import train_test_split

from openadmet.models.split.split_base import SplitterBase, splitters


@splitters.register("ShuffleSplitter")
class ShuffleSplitter(SplitterBase):
    """Vanilla splitter, uses sklearn's train_test_split which wraps ShuffleSplit."""

    def split(self, X, y):
        """
        Split the data.

        Parameters
        ----------
        X : array-like
            Feature data.
        y : array-like
            Target data.

        Returns
        -------
        tuple
            Tuple containing:
            - X_train: Training set features.
            - X_val: Validation set features (or None if val_size=0).
            - X_test: Test set features (or None if test_size=0).
            - y_train: Training set target values.
            - y_val: Validation set target values (or None if val_size=0).
            - y_test: Test set target values (or None if test_size=0).

        """
        # Training set only requested
        if self.val_size == 0 and self.test_size == 0:
            X_train, y_train = X, y
            return X_train, None, None, y_train, None, None, None

        # No test set requested
        if self.test_size == 0:
            # Split into train and val
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                train_size=None,
                test_size=int(self.val_size * X.shape[0]),
                random_state=self.random_state,
            )
            return X_train, X_val, None, y_train, y_val, None, None

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            train_size=None,
            test_size=int(self.test_size * X.shape[0]),
            random_state=self.random_state,
        )

        # No validation set requested, return train(+val) and test sets
        if self.val_size == 0:
            return X_train_val, None, X_test, y_train_val, None, y_test, None

        # Split train+val into train and val sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            train_size=None,
            test_size=int(self.val_size * X.shape[0]),
            random_state=self.random_state,
        )

        # Return train, val and test sets
        return X_train, X_val, X_test, y_train, y_val, y_test, None
