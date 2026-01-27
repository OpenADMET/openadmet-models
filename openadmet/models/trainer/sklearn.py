"""Trainers for sklearn models."""

from typing import Any

from loguru import logger
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from openadmet.models.drivers import DriverType
from openadmet.models.eval.cross_validation import (
    wrap_ktau,
    wrap_spearmanr,
)
from openadmet.models.trainer.trainer_base import TrainerBase, trainers


# Custom scorers for metrics not built into sklearn
CUSTOM_SCORERS = {
    "spearmanr": make_scorer(wrap_spearmanr),
    "ktau": make_scorer(wrap_ktau),
}


class SKLearnTrainer(TrainerBase):
    """Base trainer for sklearn models."""

    _driver_type: DriverType = DriverType.SKLEARN


@trainers.register("SKLearnBasicTrainer")
class SKlearnBasicTrainer(SKLearnTrainer):
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


class SKLearnSearchTrainer(SKLearnTrainer):
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


@trainers.register("SKLearnOptunaTrainer")
class SKLearnOptunaTrainer(SKLearnSearchTrainer):
    """
    Trainer for sklearn models with Optuna HPO.

    Performs HPO on the full dataset to produce the best production model.
    Nested CV evaluation (separate from training) provides unbiased
    performance estimates.

    Attributes
    ----------
    param_distributions : dict
        Parameter distributions for Optuna search in dict format.
    n_trials : int
        Number of Optuna trials (default: 50).
    sampler_seed : int | None
        Random seed for Optuna sampler (default: None).
    scoring : str | None
        Scoring metric for evaluation (default: None).

    """

    param_distributions: dict[str, dict[str, Any]] = {}
    n_trials: int = 50
    sampler_seed: int | None = None
    scoring: str | None = None

    def _convert_param_distributions(
        self, param_dists: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Convert parameter distributions from dict format to Optuna objects.

        Args:
            param_dists: Parameter distributions in dict format.

        Returns:
            Dictionary with Optuna distribution objects.

        """
        from optuna.distributions import (
            CategoricalDistribution,
            FloatDistribution,
            IntDistribution,
        )

        converted = {}
        for param_name, dist_config in param_dists.items():
            dist_type = dist_config["type"]
            if dist_type == "float":
                converted[param_name] = FloatDistribution(
                    low=dist_config["low"],
                    high=dist_config["high"],
                    log=dist_config.get("log", False),
                )
            elif dist_type == "int":
                converted[param_name] = IntDistribution(
                    low=dist_config["low"],
                    high=dist_config["high"],
                    log=dist_config.get("log", False),
                )
            elif dist_type == "categorical":
                converted[param_name] = CategoricalDistribution(
                    choices=dist_config["choices"]
                )
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        return converted

    def train(self, X: Any, y: Any):
        """
        Train the model with HPO on full dataset.

        Uses Optuna to find the best hyperparameters, then trains the final
        production model with those hyperparameters on the full dataset.

        Parameters
        ----------
        X : Any
            Featurized data.
        y : Any
            Target data.

        Returns
        -------
        ModelBase
            The trained model with optimized hyperparameters.

        """
        from optuna import create_study
        from optuna.integration import OptunaSearchCV  # type: ignore
        from optuna.samplers import TPESampler

        sklearn_model = self.model.estimator

        # Convert param distributions from dict to Optuna objects
        optuna_dists = self._convert_param_distributions(self.param_distributions)

        # Run Optuna HPO for production model using CV to prevent overfitting
        logger.info(f"Running Optuna HPO with CV validation (n_trials={self.n_trials})")
        sampler = (
            TPESampler(seed=self.sampler_seed)
            if self.sampler_seed is not None
            else TPESampler()
        )

        # Use negative MSE for HPO if not specified (more stable than R²)
        hpo_scoring = self.scoring if self.scoring else "neg_mean_squared_error"

        # Convert custom metric names to scorer objects
        if isinstance(hpo_scoring, str) and hpo_scoring in CUSTOM_SCORERS:
            hpo_scoring = CUSTOM_SCORERS[hpo_scoring]

        # sklearn scorers follow convention: higher is better
        # All neg_* metrics are already negated, so we always maximize
        study = create_study(sampler=sampler, direction="maximize")

        # Use simple 5-fold CV for HPO (to prevent overfitting)
        # Nested CV in evaluation provides proper performance estimation
        search = OptunaSearchCV(
            estimator=sklearn_model,
            param_distributions=optuna_dists,
            n_trials=self.n_trials,
            cv=5,  # Simple 5-fold CV to prevent overfitting during HPO
            scoring=hpo_scoring,
            study=study,
            n_jobs=1,
            verbose=0,
            return_train_score=False,
        )
        search.fit(X, y)

        best_params = search.best_params_
        logger.info(f"Best params from HPO: {best_params}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")

        # Retrain final model with best hyperparameters on full dataset
        final_model = clone(sklearn_model)
        final_model.set_params(**best_params)
        final_model.fit(X, y)

        self.search = search
        self.model.estimator = final_model
        self.model.__dict__.update(final_model.get_params())

        return self.model
