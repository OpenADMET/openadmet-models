from pydantic import BaseModel, Field
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from functools import reduce




class FeatureConcatenator(BaseModel):

    def concatenate(feats: list[ArrayLike], indices: list[np.ndarray], data: pd.DataFrame) -> np.ndarray:
        """
        Concatenate a list of features,
        """

        # use indices to mask out the features that are not present in all datasets
        common_indices = reduce(np.intersect1d, indices)
        


        # mask out failed elements from original data
        subsel_data = data.iloc[common_indices]

        raise NotImplementedError("Implement me!")