from typing import ClassVar

from modAL import ActiveLearner, CommitteeRegressor
from pydantic import Field, field_validator

from openadmet.models.active_learning.acquisition import (
    exploitation_query,
    max_uncertainty_reduction_query,
    random_query,
    upper_confidence_bound_query,
)
from openadmet.models.architecture.model_base import PickleableModelBase

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

    def train(self):
        raise NotImplementedError

    def from_params(self):
        raise NotImplementedError
