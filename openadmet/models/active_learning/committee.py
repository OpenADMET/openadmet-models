from os import PathLike
from typing import ClassVar

import numpy as np

from openadmet.models.active_learning.acquisition import _QUERY_STRATEGIES
from openadmet.models.active_learning.ensemble_base import EnsembleBase, ensemblers
from openadmet.models.architecture.model_base import ModelBase


@ensemblers.register("CommitteeRegressor")
class CommitteeRegressor(EnsembleBase):
    type: ClassVar[str] = "CommitteeRegressor"

    @classmethod
    def from_models(cls, models: list = []):
        """
        Create a committee from list of models.

        Parameters
        ----------
        models : list
            A list of committee model members.

        """

        instance = cls(
            models=models,
        )
        return instance

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
        trainer : TrainerBase
            Trainer instance, needed for deep learning models.

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
            model = mod_class(**mod_params)

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
        if query_strategy.lower() not in _QUERY_STRATEGIES:
            raise ValueError(
                f"Invalid query strategy: {query_strategy}. "
                f"Valid options are: {list(_QUERY_STRATEGIES.keys())}"
            )

        return _QUERY_STRATEGIES[query_strategy](self, X, **kwargs)

    def predict(self, X, return_std=False, **kwargs):
        """
        Make predictions using the committee model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.
        **kwargs : dict
            Additional keyword arguments to pass to the committee's predict method.

        Returns
        -------
        array-like
            Predicted values or probabilities, depending on the committee's implementation.
        """

        preds = np.stack([model.predict(X, **kwargs) for model in self.models], axis=-1)
        mean = np.mean(preds, axis=-1)
        std = np.std(preds, axis=-1)

        if return_std is True:
            return mean, std

        else:
            return mean

    def save(self, paths: list[PathLike]):
        """
        Save the committee model to the provided paths.

        Parameters
        ----------
        paths : list of PathLike
            The file paths to save the model weights.

        Returns
        -------
        None

        """

        # Check number of paths match
        if len(self.models) != len(paths):
            raise ValueError(
                "Number of models in the committee does not match the number of paths."
            )

        # Save each model to the provided paths
        for model, path in zip(self.models, paths):
            model.save(path)

    @classmethod
    def load(cls, paths: list[PathLike], models: list[ModelBase] = None):
        """
        Load a committee model from the provided paths.

        Parameters
        ----------
        paths : list of PathLike
            The file paths to the model weights.
        models : list of ModelBase
            Model instances associated with path to weights.

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
        return cls.from_models(models)

    def serialize(self, param_paths: list[PathLike], serial_paths: list[PathLike]):
        """
        Save the model to a json file and a pickled file.

        Parameters
        ----------
        param_paths : list of PathLike
            The file paths to save the model weights.
        serial_paths : list of PathLike
            The file paths to save the model architecture.

        Returns
        -------
        None

        """

        # Check number of paths match
        if len(param_paths) != len(serial_paths):
            raise ValueError("Number of parameter files and serial files do not match.")

        # Check number of models match
        if len(self.models) != len(param_paths):
            raise ValueError(
                "Number of models in the committee does not match the number of parameter files."
            )

        # Serialize each model
        for model, param_path, serial_path in zip(
            self.models, param_paths, serial_paths
        ):
            model.serialize(param_path, serial_path)

    @classmethod
    def deserialize(
        cls,
        param_paths: list[PathLike],
        serial_paths: list[PathLike],
        mod_class: ModelBase = None,
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
            raise ValueError("Number of parameter files and serial files do not match.")

        # Deserialize each model
        models = []
        for param_path, serial_path in zip(param_paths, serial_paths):
            models.append(mod_class.deserialize(param_path, serial_path))

        # Create a CommitteeRegressor instance from the deserialized models
        return cls.from_models(models)
