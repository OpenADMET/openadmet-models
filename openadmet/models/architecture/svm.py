"""Support Vector Machine (SVM) model implementations."""

from typing import ClassVar

import numpy as np
from loguru import logger
from sklearn.svm import SVC, SVR

from openadmet.models.architecture.model_base import PickleableModelBase, models


class SVMModelBase(PickleableModelBase):
    """Base class for SVM models."""

    # Meta parameters for this class
    type: ClassVar[str]
    mod_class: ClassVar[type]

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

    # Meta parameters for this class
    type: ClassVar[str] = "SVMRegressorModel"
    mod_class: ClassVar[type] = SVR

    # SVR parameters
    kernel: str = "rbf"
    degree: int = 3
    gamma: str = "scale"
    coef0: float = 0.0
    tol: float = 0.001
    C: float = 1.0
    epsilon: float = 0.1
    shrinking: bool = True
    cache_size: int = 200
    verbose: bool = False
    max_iter: int = -1


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

    # Meta parameters for this class
    type: ClassVar[str] = "SVMClassifierModel"
    mod_class: ClassVar[type] = SVC

    # SVC parameters
    C: float = 1.0
    kernel: str = "rbf"
    degree: int = 3
    gamma: str = "scale"
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = False
    tol: float = 0.001
    cache_size: int = 200
    class_weight: dict | None = None
    verbose: bool = False
    max_iter: int = -1
    decision_function_shape: str = "ovr"
    break_ties: bool = False
    random_state: int | None = None

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
