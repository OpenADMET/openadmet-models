from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from openadmet_models.eval.eval_base import EvalBase, evaluators
from openadmet_models.eval.regression import stat_and_bootstrap, nan_omit_ktau, nan_omit_spearmanr


@evaluators.register("RegressionMetrics")
class SKLearnRepeatedKFoldCrossValidation(EvalBase):
    metrics: dict = {}
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42

    _evaluated: bool = False

    def evaluate(self, model=None, X=None, y=None, **kwargs):
        """
        Evaluate the regression model
        """

        # tuple of metric, whether it is a scipy statistic, and the name to use in the report
        self.metrics = {
            "mse": (mean_squared_error, False, "MSE"),
            "mae": (mean_absolute_error, False, "MAE"),
            "r2": (r2_score, False, "$R^2$"),
            "ktau": (nan_omit_ktau, True, "Kendall's $\\tau$"),
            "spearmanr": (nan_omit_spearmanr, True, "Spearman's $\\rho$"),
        }

        # store the metric names and callables in dict suitable for sklearn cross_val_score
        self.metric_map = {k: v[0] for k, v in self.metrics.items()}

        # create a cross-validation object
        cv = RepeatedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
        )

        estimator = model.model
        # evaluate the model,
        scores = cross_val_score(estimator, X, y, cv=cv, n_jobs=-1, scoring = self.metric_map)

        print(scores)
        raise Exception("stop")
        # store the results
        self.metrics["mean_score"] = scores.mean()
        self.metrics["std_score"] = scores.std()
        self.metrics["scores"] = scores

        self._evaluated = True

        return self.metrics
    

    def write_