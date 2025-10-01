"""LightGBM model implementations."""

from typing import ClassVar

import lightgbm as lgb
import numpy as np
from loguru import logger

from openadmet.models.architecture.model_base import PickleableModelBase, models


class LGBMModelBase(PickleableModelBase):
    """Base class for LightGBM models."""

    type: ClassVar[str]
    mod_class: ClassVar[
        type
    ]  # To specify the LightGBM model class (e.g., LGBMRegressor or LGBMClassifier)
    mod_params: dict = {}

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create an LGBM model from parameters.

        Parameters
        ----------
        class_params: dict
            Parameters for the model class, such as type, mod_class, etc.
        mod_params: dict
            Parameters for the LightGBM model class, such as n_estimators, max_depth,
            learning_rate, etc.

        Returns
        -------
        instance: LGBMModelBase
            An instance of the LGBMModelBase class

        """
        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

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

    def build(self):
        """Prepare the model."""
        if not self.estimator:
            self.estimator = self.mod_class(**self.mod_params)
        else:
            logger.warning("Model already exists, skipping build")

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

    type: ClassVar[str] = "LGBMRegressorModel"
    mod_class: ClassVar[type] = lgb.LGBMRegressor


@models.register("LGBMClassifierModel")
class LGBMClassifierModel(LGBMModelBase):
    """LightGBM classification model."""

    type: ClassVar[str] = "LGBMClassifierModel"
    mod_class: ClassVar[type] = lgb.LGBMClassifier

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)
