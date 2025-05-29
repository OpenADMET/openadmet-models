from typing import ClassVar

import numpy as np
from modAL import ActiveLearner, CommitteeRegressor
from pydantic import Field, field_validator

from openadmet.models.active_learning.acquisition import (
    exploitation_query,
    max_uncertainty_reduction_query,
    random_query,
    upper_confidence_bound_query,
)
from openadmet.models.architecture.model_base import ModelBase, PickleableModelBase

_QUERY_STRATEGIES = {
    "max-uncertainty-reduction": max_uncertainty_reduction_query,
    "exploitation": exploitation_query,
    "upper-confidence-bound": upper_confidence_bound_query,  # `beta` should be configurable
    "random": random_query,
}


# @models.register("ActiveLearningCommitteeRegressor")
class ActiveLearningCommitteeRegressor(PickleableModelBase):
    """
    Committee regressor for active learning
    """

    type: ClassVar[str] = "ActiveLearningCommitteeRegressor"
    models: list = []
    query_strategy: str = Field(
        ...,
        title="Query strategy",
        description=f"The query strategy to use. Valid options are: {list(_QUERY_STRATEGIES.keys())}",
    )
    _committee: CommitteeRegressor = None

    @field_validator("query_strategy")
    @classmethod
    def validate_query_strategy(cls, value):
        """
        Validate the descriptor type
        """
        if value not in _QUERY_STRATEGIES.keys():
            raise ValueError(
                f"Query strategy {value} is not valid. "
                f"Valid options are: {list(_QUERY_STRATEGIES.keys())}"
            )
        return value

    @classmethod
    def from_models(cls, models: list = [], query_strategy: str = None):
        """
        Create a committee from list of models.
        """

        instance = cls(
            models=models,
            query_strategy=query_strategy,
        )
        instance.build()
        return instance

    @classmethod
    def train(
        cls,
        X,
        y,
        estimator: ModelBase = None,
        estimator_params: dict = {},
        n_models: int = 1,
        query_strategy: str = None,
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
        query_strategy : str, optional
            The query strategy to use.

        Returns
        -------
        ActiveLearningCommitteeRegressor
            An instance of the ActiveLearningCommitteeRegressor class.

        """

        # Verify estimator input
        if estimator is None:
            raise ValueError("Estimator must be provided.")

        # Initialize set of models
        models = []
        for i in range(n_models):
            # Initialize model
            model = estimator.from_params(mod_params=estimator_params)

            # Train
            bootstrap_idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            model.train(X[bootstrap_idx, :], y[bootstrap_idx, :])

            # Add to list
            models.append(model)

        # Instantiate the committee regressor
        return cls.from_models(models, query_strategy=query_strategy)

    def build(self):
        """
        Build the committee regressor from the list of models and query strategy.
        """

        # Map to active learners
        learners = [ActiveLearner(estimator=x) for x in self.models]

        # Assemble committee
        committee = CommitteeRegressor(
            learner_list=learners, query_strategy=_QUERY_STRATEGIES[self.query_strategy]
        )

        self._committee = committee

    def query(self, X, n_instances: int = 1, **kwargs):
        """
        Query the committee to select instances for labeling.

        Parameters
        ----------
        X : array-like
            The input data from which instances are to be queried.
        n_instances : int, optional
            The number of instances to query, by default 1.
        **kwargs : dict
            Additional keyword arguments to be passed to the committee's query method.

        Returns
        -------
        tuple
            A tuple containing the indices of the queried instances and the corresponding
            information (e.g., uncertainty scores) as determined by the committee.
        """

        return self._committee.query(X, n_instances=n_instances, **kwargs)

    def predict(self, X, **kwargs):
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

        return self._committee.predict(X, **kwargs)

    def from_params(self):
        raise NotImplementedError
