"""Committee regressor for active learning with uncertainty estimation."""

from os import PathLike
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from loguru import logger

from openadmet.models.active_learning.acquisition import _ACQUISITION_FUNCTIONS
from openadmet.models.active_learning.ensemble_base import EnsembleBase, ensemblers
from openadmet.models.architecture.model_base import ModelBase


@ensemblers.register("CommitteeRegressor")
class CommitteeRegressor(EnsembleBase):
    """
    Committee Regressor.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the ensemble model.
    _calibration_model : Any
        The calibration model used for uncertainty calibration.
    _calibration_methods : dict
        A dictionary mapping calibration method names to their corresponding functions.

    """

    type: ClassVar[str] = "CommitteeRegressor"
    _calibration_model: Any = None
    _calibration_methods: dict = {
        "isotonic-regression": "_isotonic_regression_calibration",
        "scaling-factor": "_scaling_factor_calibration",
    }

    @property
    def calibrated(self):
        """
        Check if the committee regressor has a calibration model.

        Returns
        -------
        bool
            True if the committee regressor has a calibration model, False otherwise.

        """
        return self._calibration_model is not None

    @classmethod
    def from_models(cls, models: list = []):
        """
        Create a committee from list of models.

        Parameters
        ----------
        models : list
            A list of committee model members.

        """
        # Initialize class from model list
        instance = cls(
            models=models,
        )

        return instance

    def _isotonic_regression_calibration(self, X, y, **kwargs):
        """
        Configure uncertainty calibration using isotonic regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input validation set samples to calibrate.
        y : array-like of shape (n_samples, n_features)
            The target validation set values.
        **kwargs : dict
            Additional keyword arguments to be passed to the committee's predict method.

        Returns
        -------
        None

        """
        # Reset calibration model
        self._calibration_model = None

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()

        # Predict on recalibration (validation) set
        y_pred_mean, y_pred_std = self._predict(X, return_std=True, **kwargs)

        # Fit a separate isotonic regression model for each target dimension
        calibration_models = []
        for i in range(y.shape[-1]):
            # Get the predictive uncertainties in terms of expected proportions and
            # observed proportions on the recalibration set
            y_exp_props, y_obs_props = (
                uct.metrics_calibration.get_proportion_lists_vectorized(
                    y_pred_mean[:, i], y_pred_std[:, i], y[:, i]
                )
            )

            # Train a recalibration model
            iso_model = uct.recalibration.iso_recal(y_exp_props, y_obs_props).predict

            # Append to per-dimension list
            calibration_models.append(iso_model)

        # Create per-dimension calibration model
        self._calibration_model = {"isotonic-regression": calibration_models}

    def _scaling_factor_calibration(self, X, y, **kwargs):
        """
        Configure uncertainty calibration using scaling factor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input validation set samples to calibrate.
        y : array-like of shape (n_samples, n_features)
            The target validation set values.
        **kwargs : dict
            Additional keyword arguments to be passed to the committee's predict method.

        Returns
        -------
        None

        """
        # Reset calibration model
        self._calibration_model = None

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()

        # Predict on recalibration (validation) set
        y_pred_mean, y_pred_std = self._predict(X, return_std=True, **kwargs)

        # Fit a separate scaling factor for each target dimension
        calibration_models = []
        for i in range(y.shape[-1]):
            # Determine scale factor
            scale_factor = uct.recalibration.optimize_recalibration_ratio(
                y_pred_mean[:, i], y_pred_std[:, i], y[:, i], criterion="miscal"
            )

            calibration_models.append(scale_factor)

        self._calibration_model = {"scaling-factor": calibration_models}

    def calibrate_uncertainty(self, X, y, method="isotonic-regression", **kwargs):
        """
        Configure uncertainty calibration using selected method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input validation set samples to calibrate.
        y : array-like of shape (n_samples, n_features)
            The target validation set values.
        method : str
            The calibration method to use. Options are "isotonic-regression" or "scaling-factor".
        **kwargs : dict
            Additional keyword arguments to be passed to the committee's predict method.

        Returns
        -------
        None

        """
        # Validate method selection
        if method not in self._calibration_methods:
            raise ValueError(
                f"Invalid calibration method: {method}. "
                f"Valid options are: {self._calibration_methods.keys()}."
            )

        getattr(self, self._calibration_methods[method])(X, y, **kwargs)

    def _get_calibration_function(self):
        if "scaling-factor" in self._calibration_model:
            # Create per-dimension calibration model
            return lambda x: np.stack(
                [
                    self._calibration_model["scaling-factor"][i] * (x[:, i])
                    for i in range(x.shape[-1])
                ],
                axis=1,
            )

        elif "isotonic-regression" in self._calibration_model:
            # Create per-dimension calibration model
            return lambda x: np.stack(
                [
                    self._calibration_model["isotonic-regression"][i](x[:, i])
                    for i in range(x.shape[-1])
                ],
                axis=1,
            )

    def plot_uncertainty_calibration(self, X, y, **kwargs):
        """
        Plot uncertainty calibration for the committee model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input test set samples to calibrate.
        y : array-like of shape (n_samples, n_features)
            The target test set values.
        **kwargs : dict
            Additional keyword arguments to be passed to the committee's predict method.

        Returns
        -------
        list
            A list of plots for each target dimension.

        """
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()

        # Predict on recalibration (validation) set
        y_pred_mean, y_pred_std = self.predict(X, return_std=True, **kwargs)

        # Plot calibration
        plots = []
        for i in range(y.shape[-1]):
            plots.append(
                uct.viz.plot_calibration(
                    y_pred_mean[:, i].flatten(),
                    y_pred_std[:, i].flatten(),
                    y[:, i].flatten(),
                )
            )

        # If only one plot is generated, return it directly
        if len(plots) == 1:
            return plots[0]

        return plots

    @classmethod
    def train(
        cls,
        X,
        y,
        mod_class: ModelBase = None,
        mod_params: dict = {},
        n_models: int = 1,
    ):
        """
        Train committee regressor members on bootstrapped data subsets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to train on.
        y : array-like of shape (n_samples,)
            The target values.
        mod_class : ModelBase
            The type of model to use for training.
        mod_params : dict
            The parameters to pass to the model.
        n_models : int
            The number of models in the committee, by default 1.

        Returns
        -------
        CommitteeRegressor
            An instance of the CommitteeRegressor class.

        """
        # Verify estimator input
        if mod_class is None:
            raise ValueError("Model type must be provided.")

        # Initialize set of models
        models = []
        for i in range(n_models):
            # Initialize model
            model = mod_class.from_params(mod_params=mod_params)

            # Bootstrap the data
            bootstrap_idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)

            # Train the model on the bootstrapped data
            model.train(X[bootstrap_idx, :], y[bootstrap_idx, :])

            # Add to list
            models.append(model)

        # Instantiate the committee regressor
        return cls.from_models(models)

    def query(self, X, query_strategy: str = None, **kwargs):
        """
        Query the committee to select instances for labeling.

        Parameters
        ----------
        X : array-like
            The input data from which instances are to be queried.
        query_strategy : str, optional
            The query strategy to use for selecting instances.
        **kwargs : dict
            Additional keyword arguments to be passed to the committee's query method.

        Returns
        -------
        np.array
            Values of the query strategy applied to the input data `X`.

        """
        if query_strategy.lower() not in _ACQUISITION_FUNCTIONS:
            raise ValueError(
                f"Invalid query strategy: {query_strategy}. "
                f"Valid options are: {list(_ACQUISITION_FUNCTIONS.keys())}"
            )

        mean, std = self.predict(X, return_std=True, **kwargs)

        return _ACQUISITION_FUNCTIONS[query_strategy](mean, std, **kwargs)

    def _predict(self, X, return_std=False, **kwargs):
        """
        Make predictions using the committee model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.
        return_std : bool, optional
            Whether to return the standard deviation of the predictions.
        **kwargs : dict
            Additional keyword arguments to pass to the committee's predict method.

        Returns
        -------
        array-like
            Predicted values or probabilities, depending on the committee's implementation.

        """
        # Make predictions
        preds = np.stack([model.predict(X, **kwargs) for model in self.models], axis=-1)

        # Compute mean
        mean = np.mean(preds, axis=-1)

        # Skip std if not requested
        if return_std is False:
            return mean

        # Compute standard deviation
        std = np.std(preds, axis=-1)

        # Calibrate std if calibration model is available
        if self.calibrated:
            std = self._get_calibration_function()(std)

        return mean, std

    def predict(self, X, return_std=False, **kwargs):
        """
        Make predictions using the committee model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.
        return_std : bool, optional
            Whether to return the standard deviation of the predictions.
        **kwargs : dict
            Additional keyword arguments to pass to the committee's predict method.

        Returns
        -------
        array-like
            Predicted values or probabilities, depending on the committee's implementation.

        """
        if return_std is True and not self.calibrated:
            logger.warning(
                "Standard deviation not calibrated: consider calling `calibrate_uncertainty`."
            )

        return self._predict(X, return_std=return_std, **kwargs)

    def _save_calibration_model(self, path: PathLike = "calibration_model.pkl"):
        # Save calibration model
        if self.calibrated:
            with open(path, "wb") as f:
                joblib.dump(self._calibration_model, f)

        else:
            logger.warning(
                "Standard deviation not calibrated: consider calling `calibrate_uncertainty` before saving."
            )

    def _load_calibration_model(self, path: PathLike = "calibration_model.pkl"):
        # Load calibration model
        with open(path, "rb") as f:
            self._calibration_model = joblib.load(f)

        logger.info(f"Successfully loaded calibration from {path}")

    def save(
        self,
        paths: list[PathLike],
        calibration_path: PathLike = "calibration_model.pkl",
    ):
        """
        Save the committee model to the provided paths.

        Parameters
        ----------
        paths : list of PathLike
            The file paths to save the model weights.
        calibration_path: PathLike
            Path to save calibration model.

        Returns
        -------
        None

        """
        # Check number of paths match
        if self.n_models != len(paths):
            raise ValueError(
                f"Number of models ({self.n_models}) in the committee does not match the number of paths ({len(paths)})."
            )

        # Save each model to the provided paths
        for model, path in zip(self.models, paths):
            model.save(path)

        # Save calibration model
        self._save_calibration_model(calibration_path)

    @classmethod
    def load(
        cls,
        paths: list[PathLike],
        models: list[ModelBase] = None,
        calibration_path: PathLike = None,
    ):
        """
        Load a committee model from the provided paths.

        Parameters
        ----------
        paths : list of PathLike
            The file paths to the model weights.
        models : list of ModelBase
            Model instances associated with path to weights.
        calibration_path : PathLike
            The file path to the calibration model.

        Returns
        -------
        CommitteeRegressor
            A committee model created from the loaded models.

        """
        # Check model type
        if models is None:
            raise ValueError("Must provide a list of model instances to load.")

        # Check lengths match
        if len(paths) != len(models):
            raise ValueError("Number of paths and models do not match.")

        # Load each model from the provided paths
        [model.load(path) for model, path in zip(models, paths)]

        # Create a CommitteeRegressor instance from the loaded models
        instance = cls.from_models(models)

        # Load calibration model
        if calibration_path is not None:
            instance._load_calibration_model(calibration_path)

        return instance

    def serialize(
        self,
        param_paths: list[PathLike],
        serial_paths: list[PathLike],
        calibration_path: PathLike = "calibration_model.pkl",
    ):
        """
        Save the model to json files and pickled files.

        Parameters
        ----------
        param_paths : list of PathLike
            The file paths to save the model weights.
        serial_paths : list of PathLike
            The file paths to save the model architecture.
        calibration_path : PathLike
            The file path to save the calibration model.

        Returns
        -------
        None

        """
        # Check number of paths match
        if len(param_paths) != len(serial_paths):
            raise ValueError(
                f"Number of parameter files ({len(param_paths)}) and serial files ({len(serial_paths)}) do not match."
            )

        # Check number of models match
        if self.n_models != len(param_paths):
            raise ValueError(
                f"Number of models ({self.n_models}) in the committee does not match the number of parameter files ({len(param_paths)})."
            )

        # Serialize each model
        for model, param_path, serial_path in zip(
            self.models, param_paths, serial_paths
        ):
            model.serialize(param_path, serial_path)

        # Save calibration model
        self._save_calibration_model(calibration_path)

    @classmethod
    def deserialize(
        cls,
        param_paths: list[PathLike],
        serial_paths: list[PathLike],
        mod_class: ModelBase = None,
        calibration_path: PathLike = None,
    ):
        """
        Create a model from parameters and a pickled model.

        Parameters
        ----------
        param_paths : list of PathLike
            The file paths to the model parameters.
        serial_paths : list of PathLike
            The file paths to the model serializations.
        mod_class : ModelBase
            Model class to update with the deserialized parameters.
        calibration_path : PathLike
            The file path to the calibration model.

        Returns
        -------
        Committee
            A committee model created from the deserialized parameters.

        """
        # Check model type
        if mod_class is None:
            raise ValueError("Must provide a model type to load.")

        # Check lengths match
        if len(param_paths) != len(serial_paths):
            raise ValueError(
                f"Number of parameter files {len(param_paths)} and serial files {len(serial_paths)} do not match."
            )

        # Deserialize each model
        models = []
        for param_path, serial_path in zip(param_paths, serial_paths):
            models.append(mod_class.deserialize(param_path, serial_path))

        # Create a CommitteeRegressor instance from the deserialized models
        instance = cls.from_models(models)

        # Load calibration model
        if calibration_path is not None:
            instance._load_calibration_model(calibration_path)

        return instance
