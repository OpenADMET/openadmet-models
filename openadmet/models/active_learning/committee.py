from typing import ClassVar

import numpy as np

from openadmet.models.active_learning.acquisition import _QUERY_STRATEGIES
from openadmet.models.architecture.model_base import ModelBase, PickleableModelBase


# @models.register("CommitteeRegressor")
class CommitteeRegressor(PickleableModelBase):
    type: ClassVar[str] = "CommitteeRegressor"
    models: list = []

    def build(self):
        pass

    @classmethod
    def from_models(cls, models: list = []):
        """
        Create a committee from list of models.
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
        estimator: ModelBase = None,
        estimator_params: dict = {},
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
        estimator : ModelBase
            The type of model to use for training.
        estimator_params : dict
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
        if estimator is None:
            raise ValueError("Estimator must be provided.")

        # Initialize set of models
        models = []
        for i in range(n_models):
            # Initialize model
            model = estimator.from_params(mod_params=estimator_params)

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

    def predict(self, X, return_std=False):
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

        preds = np.stack([model.predict(X) for model in self.models], axis=-1)
        mean = np.mean(preds, axis=-1)
        std = np.std(preds, axis=-1)

        if return_std is True:
            return mean, std

        else:
            return mean

    def from_params(self):
        """
        This method doesn't really make sense for this class, as it is instantiated from already-trained models
        or from the `train` method.
        """
        raise NotImplementedError
