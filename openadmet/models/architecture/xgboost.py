from typing import ClassVar

from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from loguru import logger

from openadmet.models.architecture.model_base import PickleableModelBase, models


class XGBoostModelBase(PickleableModelBase):
    """
    Base class for XGBoost models
    """

    type: ClassVar[str]
    mod_class: ClassVar[
        type
    ]  # To specify the XGBoost model class (e.g., XGBMRegressor or XGBMClassifier)
    mod_params: dict = {}

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters
        """
        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model
        """
        self.build()
        self.estimator = self.estimator.fit(X, y, verbose=True)

    def build(self):
        """
        Prepare the model
        """
        if not self.estimator:
            self.estimator = self.mod_class(**self.mod_params)
        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("XGBRegressorModel")
class XGBRegressorModel(XGBoostModelBase):
    """
    LightGBM regression model
    """

    type: ClassVar[str] = "XGBRegressorModel"
    mod_class: ClassVar[type] = XGBRegressor


@models.register("XGBClassifierModel")
class XGBClassifierModel(XGBoostModelBase):
    """
    LightGBM classification model
    """

    type: ClassVar[str] = "XGBoostClaXGBClassifierModelssifierModel"
    mod_class: ClassVar[type] = XGBClassifier

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)
