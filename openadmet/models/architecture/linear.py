"""Linear model implementations for regression and classification."""

import json
from os import PathLike
from typing import ClassVar

import joblib
import numpy as np
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from openadmet.models.architecture.model_base import PickleableModelBase, models


class LinearModelBase(PickleableModelBase):
    """Base class for sklearn linear models."""

    # Meta parameters for this class
    type: ClassVar[str]
    mod_class: ClassVar[type]  # type: ignore

    # Imputation parameter
    use_mean_imputation: bool = False

    @property
    def _imputer(self):
        """
        Get the imputer from the pipeline if it exists.

        Returns
        -------
        SimpleImputer or None
            The imputer if the model uses a pipeline with imputation,
            None otherwise.

        """
        if isinstance(self.estimator, Pipeline):
            return self.estimator.named_steps.get("imputer")
        return None

    def build(self):
        """
        Prepare the model.

        If use_mean_imputation is True, wraps the estimator in a Pipeline
        with SimpleImputer and StandardScaler. Otherwise, uses the estimator
        directly.
        """
        if not self.estimator:
            model_params = self.model_dump(exclude={"use_mean_imputation"})
            base_model = self.mod_class(**model_params)

            if self.use_mean_imputation:
                # Wrap in pipeline with imputer and scaler
                # StandardScaler is important for linear models with
                # regularization
                self.estimator = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                        ("model", base_model),
                    ]
                )
            else:
                self.estimator = base_model
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
        self.estimator = self.estimator.fit(X, y)

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

    def save(self, path: PathLike):
        """
        Save the model to a pickle file.

        Parameters
        ----------
        path: PathLike
            Path to save the model to

        """
        if self.estimator is None:
            raise ValueError("Model is not built, cannot save")

        with open(path, "wb") as f:
            joblib.dump(self.estimator, f)

    def load(self, path: PathLike):
        """
        Load the model from a pickle file.

        Parameters
        ----------
        path: PathLike
            Path to load the model from

        """
        with open(path, "rb") as f:
            self.estimator = joblib.load(f)

    @classmethod
    def deserialize(
        cls,
        param_path: PathLike = "model.json",
        serial_path: PathLike = "model.pkl",
    ):
        """
        Create a model from parameters and a pickled model.

        Parameters
        ----------
        param_path: PathLike
            Path to load the model parameters from
        serial_path: PathLike
            Path to load the pickled model from

        Returns
        -------
        instance: LinearModelBase
            An instance of the LinearModelBase class

        """
        with open(param_path) as f:
            mod_params = json.load(f)
        instance = cls(**mod_params)
        instance.build()
        instance.load(serial_path)
        return instance


@models.register("RidgeModel")
class RidgeModel(LinearModelBase):
    """Ridge regression model (L2 regularization)."""

    # Meta parameters for this class
    type: ClassVar[str] = "RidgeModel"
    mod_class: ClassVar[type] = Ridge  # type: ignore

    # Ridge parameters
    alpha: float = 1.0
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: int | None = None
    tol: float = 0.0001
    solver: str = "auto"
    positive: bool = False
    random_state: int | None = None


@models.register("LassoModel")
class LassoModel(LinearModelBase):
    """Lasso regression model (L1 regularization)."""

    # Meta parameters for this class
    type: ClassVar[str] = "LassoModel"
    mod_class: ClassVar[type] = Lasso  # type: ignore

    # Lasso parameters
    alpha: float = 1.0
    fit_intercept: bool = True
    precompute: bool = False
    copy_X: bool = True
    max_iter: int = 1000
    tol: float = 0.0001
    warm_start: bool = False
    positive: bool = False
    random_state: int | None = None
    selection: str = "cyclic"


@models.register("ElasticNetModel")
class ElasticNetModel(LinearModelBase):
    """ElasticNet regression model (L1 + L2 regularization)."""

    # Meta parameters for this class
    type: ClassVar[str] = "ElasticNetModel"
    mod_class: ClassVar[type] = ElasticNet  # type: ignore

    # ElasticNet parameters
    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    precompute: bool = False
    max_iter: int = 1000
    copy_X: bool = True
    tol: float = 0.0001
    warm_start: bool = False
    positive: bool = False
    random_state: int | None = None
    selection: str = "cyclic"


class LogisticRegressionBase(PickleableModelBase):
    """Base class for logistic regression models."""

    # Meta parameters for this class
    type: ClassVar[str]
    mod_class: ClassVar[type] = LogisticRegression  # type: ignore

    # Imputation parameter
    use_mean_imputation: bool = False

    @property
    def _imputer(self):
        """
        Get the imputer from the pipeline if it exists.

        Returns
        -------
        SimpleImputer or None
            The imputer if the model uses a pipeline with imputation,
            None otherwise.

        """
        if isinstance(self.estimator, Pipeline):
            return self.estimator.named_steps.get("imputer")
        return None

    def build(self):
        """
        Prepare the model.

        If use_mean_imputation is True, wraps the estimator in a Pipeline
        with SimpleImputer and StandardScaler. Otherwise, uses the estimator
        directly.
        """
        if not self.estimator:
            model_params = self.model_dump(exclude={"use_mean_imputation"})
            base_model = self.mod_class(**model_params)

            if self.use_mean_imputation:
                # Wrap in pipeline with imputer and scaler
                # StandardScaler is important for linear models with
                # regularization
                self.estimator = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                        ("model", base_model),
                    ]
                )
            else:
                self.estimator = base_model
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
        self.estimator = self.estimator.fit(X, y)

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

    def save(self, path: PathLike):
        """
        Save the model to a pickle file.

        Parameters
        ----------
        path: PathLike
            Path to save the model to

        """
        if self.estimator is None:
            raise ValueError("Model is not built, cannot save")

        with open(path, "wb") as f:
            joblib.dump(self.estimator, f)

    def load(self, path: PathLike):
        """
        Load the model from a pickle file.

        Parameters
        ----------
        path: PathLike
            Path to load the model from

        """
        with open(path, "rb") as f:
            self.estimator = joblib.load(f)

    @classmethod
    def deserialize(
        cls,
        param_path: PathLike = "model.json",
        serial_path: PathLike = "model.pkl",
    ):
        """
        Create a model from parameters and a pickled model.

        Parameters
        ----------
        param_path: PathLike
            Path to load the model parameters from
        serial_path: PathLike
            Path to load the pickled model from

        Returns
        -------
        instance: LogisticRegressionBase
            An instance of the LogisticRegressionBase class

        """
        with open(param_path) as f:
            mod_params = json.load(f)
        instance = cls(**mod_params)
        instance.build()
        instance.load(serial_path)
        return instance


@models.register("LogisticRegressionL1Model")
class LogisticRegressionL1Model(LogisticRegressionBase):
    """Logistic regression with L1 regularization."""

    # Meta parameters for this class
    type: ClassVar[str] = "LogisticRegressionL1Model"

    # Logistic Regression parameters
    penalty: str = "l1"
    dual: bool = False
    tol: float = 0.0001
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: dict | str | None = None
    random_state: int | None = None
    solver: str = "liblinear"
    max_iter: int = 100
    multi_class: str = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: int | None = None
    l1_ratio: float | None = None


@models.register("LogisticRegressionL2Model")
class LogisticRegressionL2Model(LogisticRegressionBase):
    """Logistic regression with L2 regularization."""

    # Meta parameters for this class
    type: ClassVar[str] = "LogisticRegressionL2Model"

    # Logistic Regression parameters
    penalty: str = "l2"
    dual: bool = False
    tol: float = 0.0001
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: dict | str | None = None
    random_state: int | None = None
    solver: str = "lbfgs"
    max_iter: int = 100
    multi_class: str = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: int | None = None
    l1_ratio: float | None = None


@models.register("LogisticRegressionElasticNetModel")
class LogisticRegressionElasticNetModel(LogisticRegressionBase):
    """Logistic regression with ElasticNet (L1 + L2) regularization."""

    # Meta parameters for this class
    type: ClassVar[str] = "LogisticRegressionElasticNetModel"

    # Logistic Regression parameters
    penalty: str = "elasticnet"
    dual: bool = False
    tol: float = 0.0001
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    class_weight: dict | str | None = None
    random_state: int | None = None
    solver: str = "saga"
    max_iter: int = 100
    multi_class: str = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: int | None = None
    l1_ratio: float = 0.5
