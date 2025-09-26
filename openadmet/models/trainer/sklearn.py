"""Trainers for sklearn models."""

from typing import Any

from loguru import logger
from sklearn.model_selection import GridSearchCV

from openadmet.models.trainer.trainer_base import TrainerBase, trainers


@trainers.register("SKLearnBasicTrainer")
class SKlearnBasicTrainer(TrainerBase):
    """Basic trainer for sklearn models."""

    def train(self, X: Any, y: Any):
        """
        Train the model.

        Parameters
        ----------
        X : Any
            Feature data.
        y : Any
            Target data.

        Returns
        -------
        ModelBase
            The trained model.

        """
        sklearn_model = self.model.estimator
        sklearn_model.fit(X, y)
        self.model.estimator = sklearn_model
        return self.model

    def build(self):
        """Unused method for sklearn models."""
        pass


class SKLearnSearchTrainer(TrainerBase):
    """
    Trainer for sklearn models with search.

    Attributes
    ----------
    search : Any
        The search object (e.g., GridSearchCV).

    """

    _search: Any

    @property
    def search(self):
        """Return search object (e.g., GridSearchCV)."""
        return self._search

    @search.setter
    def search(self, value):
        """Set search object (e.g., GridSearchCV)."""
        self._search = value

    def build(self):
        """Unused method for sklearn models."""
        pass


@trainers.register("SKLearnGridSearchTrainer")
class SKLearnGridSearchTrainer(SKLearnSearchTrainer):
    """
    Trainer for sklearn models with grid search.

    Attributes
    ----------
    param_grid : dict
        The parameter grid for grid search.

    """

    param_grid: dict = {}

    def train(self, X: Any, y: Any):
        """
        Train the model.

        Parameters
        ----------
        X : Any
            Featurized data.
        y : Any
            Target data.

        Returns
        -------
        ModelBase
            The trained model.

        """
        sklearn_model = self.model.estimator
        self.search = GridSearchCV(sklearn_model, param_grid=self.param_grid)
        self.search.fit(X, y)

        # Set the params and model to the best found
        self.model.estimator = self.search.best_estimator_
        self.model.__dict__.update(self.model.estimator.get_params())

        logger.info(f"Best params: {self.model.estimator.get_params()}")
        return self.model
