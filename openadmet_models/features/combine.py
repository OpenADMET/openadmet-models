from pydantic import BaseModel, Field
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike



class FeatureConcatenator(BaseModel):

    def concatenate(feats: list[ArrayLike], indexes: list[np.ndarray], data: pd.DataFrame) -> np.ndarray:
        """
        Concatenate a list of features,
        """
        pass
