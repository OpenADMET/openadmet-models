from typing import ClassVar

from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from loguru import logger

from openadmet.models.architecture.model_base import PickleableModelBase, models


class XGBoostModelBase(PickleableModelBase):
    """
    Base class for XGBoost models, allows instantiation from parameters that are passable to the XGBoost model classes.
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
    XGBoost regression model

    Common parameters for XGBoost models can be found at:
    https://xgboost.readthedocs.io/en/stable/python/python_api.html

    Common parameters that you might want to set include:
    - n_estimators: Number of trees in the ensemble
    - max_depth: Maximum depth of a tree
    - max_leaves: Maximum number of leaves in a tree
    - learning_rate: Step size shrinkage used in update to prevent overfitting
    - objective: Specify the learning task and corresponding objective function
    - booster: Specify which booster to use, options are gbtree, gblinear or dart
    - tree_method: Specify the tree construction algorithm used in XGBoost
    """

    type: ClassVar[str] = "XGBRegressorModel"
    mod_class: ClassVar[type] = XGBRegressor


@models.register("XGBClassifierModel")
class XGBClassifierModel(XGBoostModelBase):
    """
    XGBoost classification model

    Common parameters for XGBoost models can be found at:
    https://xgboost.readthedocs.io/en/stable/python/python_api.html
    Common parameters that you might want to set include:
    - n_estimators: Number of trees in the ensemble
    - max_depth: Maximum depth of a tree
    - max_leaves: Maximum number of leaves in a tree
    - learning_rate: Step size shrinkage used in update to prevent overfitting
    - objective: Specify the learning task and corresponding objective function
    - booster: Specify which booster to use, options are gbtree, gblinear or dart
    - tree_method: Specify the tree construction algorithm used in XGBoost
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
