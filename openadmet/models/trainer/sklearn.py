from typing import Any

from loguru import logger
from sklearn.model_selection import GridSearchCV

from openadmet.models.trainer.trainer_base import TrainerBase, trainers


@trainers.register("SKLearnBasicTrainer")
class SKlearnBasicTrainer(TrainerBase):
    """
    Basic trainer for sklearn models
    """

    def train(self, X: Any, y: Any):
        sklearn_model = self.model.estimator
        sklearn_model.fit(X, y)
        self.model.estimator = sklearn_model
        return self.model


class SKLearnSearchTrainer(TrainerBase):
    """
    Trainer for sklearn models with search
    """

    _search: Any

    @property
    def search(self):
        return self._search

    @search.setter
    def search(self, value):
        self._search = value


@trainers.register("SKLearnGridSearchTrainer")
class SKLearnGridSearchTrainer(SKLearnSearchTrainer):
    """
    Trainer for sklearn models with grid search
    """

    param_grid: dict = {}

    def train(self, X: Any, y: Any):
        """
        Train the model
        """
        sklearn_model = self.model.estimator
        self.search = GridSearchCV(sklearn_model, param_grid=self.param_grid)
        self.search.fit(X, y)
        # set the params and model to the best found
        self.model.estimator = self.search.best_estimator_
        self.model.model_params = self.model.estimator.get_params()
        logger.info(f"Best params: {self.model.model_params}")
        return self.model
