"""Support Vector Machine (SVM) model implementations."""

from typing import ClassVar

from sklearn.svm import SVR, SVC
import numpy as np
from loguru import logger

from openadmet.models.architecture.model_base import PickleableModelBase, models


class SVMModelBase(PickleableModelBase):
    """Base class for SVM models."""

    type: ClassVar[str]
    mod_class: ClassVar[
        type
    ]  # To specify the XGBoost model class (e.g., XGBMRegressor or XGBMClassifier)
    mod_params: dict = {}

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters.

        Parameters
        ----------
        class_params: dict
            Parameters for the model class, such as type, mod_class, etc.
        mod_params: dict
            Parameters for the XGBoost model class, such as n_estimators, max_depth,
            learning_rate, etc.

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
            Training data labels

        """
        self.build()
        self.estimator = self.estimator.fit(X, y, verbose=True)

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
            Data to predict on
        **kwargs
            Additional keyword arguments for the predict method.

        Returns
        -------
        np.ndarray
            Predictions from the model

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("SVMRegressorModel")
class SVMRegressorModel(SVMModelBase):
    """
    SVM regression model.

    Common parameters for SVM models can be found at:
    https://scikit-learn.org/stable/modules/svm.html

    Common parameters that you might want to set include:
    - C: Regularization parameter
    - kernel: Specifies the kernel type to be used in the algorithm
    - degree: Degree of the polynomial kernel function (if using 'poly' kernel)
    - learning_rate: Step size shrinkage used in update to prevent overfitting
    - objective: Specify the learning task and corresponding objective function
    - booster: Specify which booster to use, options are gbtree, gblinear or dart
    - tree_method: Specify the tree construction algorithm used in XGBoost
    """

    type: ClassVar[str] = "SVMRegressorModel"
    mod_class: ClassVar[type] = SVR


@models.register("SVMClassifierModel")
class SVMClassifierModel(SVMModelBase):
    """
    SVM classification model.

    Common parameters for SVM models can be found at:
    https://scikit-learn.org/stable/modules/svm.html
    Common parameters that you might want to set include:
    - C: Regularization parameter
    - kernel: Specifies the kernel type to be used in the algorithm
    - degree: Degree of the polynomial kernel function (if using 'poly' kernel)
    - learning_rate: Step size shrinkage used in update to prevent overfitting
    - objective: Specify the learning task and corresponding objective function
    - booster: Specify which booster to use, options are gbtree, gblinear or dart
    - tree_method: Specify the tree construction algorithm used in XGBoost
    """

    type: ClassVar[str] = "SVMClassifierModel"
    mod_class: ClassVar[type] = SVC

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
