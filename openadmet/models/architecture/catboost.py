"""CatBoost model implementations."""

from typing import ClassVar

import numpy as np
from loguru import logger
from pydantic import ConfigDict

from openadmet.models._seed import RandomSeedMixin
from openadmet.models.architecture.model_base import PickleableModelBase, models


class CatBoostModelBase(RandomSeedMixin, PickleableModelBase):
    """
    Base class for CatBoost models, allows instantiation from parameters that are passable to the CatBoost model classes.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the model.

    """

    # Allow extra arguments
    model_config = ConfigDict(extra="allow")

    # Meta parameters for this class
    type: ClassVar[str]

    @classmethod
    def _get_estimator_class(cls) -> type:
        """Return the CatBoost estimator class (deferred import)."""
        raise NotImplementedError

    def build(self):
        """Prepare the model."""
        if not self.estimator:
            self.estimator = self._get_estimator_class()(**self.model_dump())
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
            Training data labels

        """
        self.build()
        self.estimator = self.estimator.fit(X, y, verbose=True)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X: np.ndarray
            Data to predict on
        kwargs: dict
            Additional keyword arguments to pass to the predict method of the CatBoost model

        Returns
        -------
        np.ndarray
            Predictions from the model

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("CatBoostRegressorModel")
class CatBoostRegressorModel(CatBoostModelBase):
    """
    CatBoost regression model.

    Common parameters for CatBoost models can be found at:
    https://catboost.ai/docs/en/concepts/python-quickstart

    Common parameters that you might want to set include:
    - n_estimators: Number of trees in the ensemble
    - max_depth: Maximum depth of a tree
    - max_leaves: Maximum number of leaves in a tree
    - learning_rate: Step size shrinkage used in update to prevent overfitting
    - objective: Specify the learning task and corresponding objective function
    - booster: Specify which booster to use, options are gbtree, gblinear or dart
    - tree_method: Specify the tree construction algorithm used in CatBoost
    """

    # Meta parameters for this class
    type: ClassVar[str] = "CatBoostRegressorModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from catboost import CatBoostRegressor

        return CatBoostRegressor


@models.register("CatBoostClassifierModel")
class CatBoostClassifierModel(CatBoostModelBase):
    """
    CatBoost classification model.

    Common parameters for CatBoost models can be found at:
    https://catboost.ai/docs/en/concepts/python-quickstart
    """

    # Meta parameters for this class
    type: ClassVar[str] = "CatBoostClassifierModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from catboost import CatBoostClassifier

        return CatBoostClassifier

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model, returning probabilities for each class.

        Parameters
        ----------
        X: np.ndarray
            Data to predict on

        Returns
        -------
        np.ndarray
            Probabilities for each class from the model

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)
