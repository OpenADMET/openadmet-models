from splito import MaxDissimilaritySplit, PerimeterSplit, ScaffoldSplit

from openadmet_models.split.split_base import SplitterBase, splitters


@splitters.register("ScaffoldSplitter")
class ScaffoldSplitter(SplitterBase):
    """
    Splits the data based on the scaffold of the molecules
    """

    def split(self, X, Y):
        """
        Split the data
        """
        splitter = ScaffoldSplit(
            smiles=X,
            n_jobs=-1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_idx, test_idx = next(splitter.split(X=X))
        return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


@splitters.register("PerimeterSplitter")
class PerimeterSplitter(SplitterBase):
    """ """

    def split(self, X, Y):
        """
        Split the data
        """
        splitter = PerimeterSplit(
            n_jobs=-1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, test_idx = next(splitter.split(X=X))
        return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


@splitters.register("MaxDissimilaritySplitter")
class MaxDissimilaritySplitter(SplitterBase):
    """ """

    def split(self, X, Y):
        """
        Split the data
        """
        splitter = MaxDissimilaritySplit(
            n_jobs=-1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, test_idx = next(splitter.split(X=X))
        return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]
