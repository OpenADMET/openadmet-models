from sklearn.model_selection import train_test_split

from openadmet_models.split.split_base import SplitterBase, splitters


@splitters.register("ShuffleSplitter")
class ShuffleSplitter(SplitterBase):
    """
    Vanilla splitter, uses sklearn's train_test_split which wraps ShuffleSplit
    """

    def split(self, X, Y):
        """
        Split the data
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            Y,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test
