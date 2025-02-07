

from openadmet_models.split.split_base import SplitterBase, splitters
from splito import ScaffoldSplit


@splitters.register("ScaffoldSplitter")
class ShuffleSplitter(SplitterBase):
    """
    Vanilla splitter, uses sklearn's train_test_split which wraps ShuffleSplit
    """

    def split(self, X, Y):
        """
        Split the data
        """
        splitter = ScaffoldSplit(smiles=X, n_jobs=-1, test_size=self.test_size, random_state=self.random_state)
        train_idx, test_idx = next(splitter.split(X=X))
        return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

