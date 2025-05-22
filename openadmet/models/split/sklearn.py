from sklearn.model_selection import train_test_split

from openadmet.models.split.split_base import SplitterBase, splitters


@splitters.register("ShuffleSplitter")
class ShuffleSplitter(SplitterBase):
    """
    Vanilla splitter, uses sklearn's train_test_split which wraps ShuffleSplit
    """

    def split(self, X, y):
        """
        Split the data
        """

        # First split into train and val+test
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X,
            y,
            train_size=None,
            test_size=int((self.test_size + self.val_size) * X.shape[0]),
            random_state=self.random_state,
        )

        # If no validation set is requested, return train and test sets
        if self.val_size == 0:
            return X_train, None, X_val_test, y_train, None, y_val_test

        # If no test set is requested, return train and validation sets
        elif self.test_size == 0:
            return X_train, X_val_test, None, y_train, y_val_test, None

        # If both test and validation sets are requested, split the remaining data
        else:
            X_val, X_test, y_val, y_test = train_test_split(
                X_val_test,
                y_val_test,
                train_size=None,
                test_size=int(self.test_size * X.shape[0]),
                random_state=self.random_state,
            )

            return X_train, X_val, X_test, y_train, y_val, y_test
