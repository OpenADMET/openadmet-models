"""LightGBM model implementations."""

from typing import ClassVar

import lightgbm as lgb
import numpy as np
from loguru import logger

from openadmet.models.architecture.model_base import PickleableModelBase, models


class LGBMModelBase(PickleableModelBase):
    """Base class for LightGBM models."""

    # Meta parameters for this class
    type: ClassVar[str]
    mod_class: ClassVar[type]

    # LGBM parameters
    boosting_type: str = "gbdt"
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample_for_bin: int = 200000
    objective: str | None = None
    class_weight: str | None = None
    min_split_gain: float = 0.0
    min_child_weight: float = 0.001
    min_child_samples: int = 20
    subsample: float = 1.0
    subsample_freq: int = 0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int | None = None
    n_jobs: int | None = None
    importance_type: str = "split"
    verbose: int = -1

    def build(self):
        """Prepare the model."""
        if not self.estimator:
            self.estimator = self.mod_class(**self.model_dump())
        else:
            logger.warning("Model already exists, skipping build")

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.

        Parameters
        ----------
        X: np.ndarray
            Training data features
        y: np.ndarray
            Training data values

        """
        self.build()
        self.estimator = self.estimator.fit(X, y)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X: np.ndarray
            Featurized data to predict on
        kwargs: dict
            Additional keyword arguments to pass to the predict method of the LightGBM model

        Returns
        -------
        np.ndarray
            Predictions from the model

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("LGBMRegressorModel")
class LGBMRegressorModel(LGBMModelBase):
    """LightGBM regression model."""

    # Meta parameters for this class
    type: ClassVar[str] = "LGBMRegressorModel"
    mod_class: ClassVar[type] = lgb.LGBMRegressor


@models.register("LGBMClassifierModel")
class LGBMClassifierModel(LGBMModelBase):
    """LightGBM classification model."""

    # Meta parameters for this class
    type: ClassVar[str] = "LGBMClassifierModel"
    mod_class: ClassVar[type] = lgb.LGBMClassifier

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)
