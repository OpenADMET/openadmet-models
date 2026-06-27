"""Sklearn-based data splitting implementations."""

import numpy as np
from sklearn.model_selection import train_test_split

from openadmet.models.split.split_base import SplitterBase, splitters


def _take(obj, idx):
    """Positionally select rows from a pandas or numpy object."""
    return obj.iloc[idx] if hasattr(obj, "iloc") else obj[idx]


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
                random_state=self.random_seed,
            )
            return X_train, X_val, None, y_train, y_val, None, None

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            train_size=None,
            test_size=int(self.test_size * X.shape[0]),
            random_state=self.random_seed,
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
            random_state=self.random_seed,
        )

        # Return train, val and test sets
        return X_train, X_val, X_test, y_train, y_val, y_test, None


@splitters.register("TailFractionSplitter")
class TailFractionSplitter(ShuffleSplitter):
    """
    ShuffleSplitter that keeps only a fraction of the high-target training rows.

    Reproduces the parent ShuffleSplitter partition exactly (same ``random_seed`` and
    sizes), so the validation and test sets are identical to a plain ShuffleSplitter run
    and ``tail_fraction=1.0`` is the unmodified baseline. It then drops a seeded fraction
    of the training rows whose target is at or above ``tail_threshold``, leaving the
    sub-threshold training rows and the val/test sets untouched. This isolates the
    marginal value of potent training examples for a tail learning curve: vary
    ``tail_fraction`` across runs and the test set stays fixed.

    Attributes
    ----------
    tail_threshold : float
        Target value at or above which a training row counts as a potent tail example.
    tail_fraction : float
        Fraction of the potent training rows to keep, in [0, 1]. 1.0 keeps all (baseline);
        0.0 removes every potent training row.
    subsample_seed : int
        Seed for the random choice of which potent training rows to keep, so the
        subsample is reproducible and independent of the split seed.

    """

    tail_threshold: float = 6.0
    tail_fraction: float = 1.0
    subsample_seed: int = 0

    def split(self, X, y):
        """Split as ShuffleSplitter, then subsample potent rows from the training set only."""
        X_train, X_val, X_test, y_train, y_val, y_test, groups = super().split(X, y)
        if self.tail_fraction >= 1.0:
            return X_train, X_val, X_test, y_train, y_val, y_test, groups

        # locate potent training rows by the (single) target column, then keep a seeded
        # fraction of them alongside every sub-threshold row
        y_values = y_train.iloc[:, 0] if hasattr(y_train, "columns") else y_train
        y_values = np.asarray(y_values).ravel()
        potent = np.flatnonzero(y_values >= self.tail_threshold)
        n_keep = int(round(self.tail_fraction * len(potent)))
        rng = np.random.default_rng(self.subsample_seed)
        kept_potent = rng.choice(potent, size=n_keep, replace=False) if n_keep else np.array([], dtype=int)
        keep = np.sort(np.concatenate([np.flatnonzero(y_values < self.tail_threshold), kept_potent]))

        return _take(X_train, keep), X_val, X_test, _take(y_train, keep), y_val, y_test, groups
