from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from openadmet_models.eval.eval_base import EvalBase, evaluators
from openadmet_models.eval.regression import stat_and_bootstrap, nan_omit_ktau, nan_omit_spearmanr


@evaluators.register("SKLearnRepeatedKFoldCrossValidation")
class SKLearnRepeatedKFoldCrossValidation(EvalBase):
    metrics: dict = {}
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42

    _evaluated: bool = False

    def evaluate(self, model=None, X_train=None, y_train=None, **kwargs):
        """
        Evaluate the regression model
        """
        if model is None or X_train is None or y_train is None:
            raise ValueError("model, X_train, and y_train must be provided")

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
        print(self.metric_map)

        # create a cross-validation object
        cv = RepeatedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
        )

        estimator = model.model
        # evaluate the model,
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, n_jobs=-1, scoring = self.metric_map)

        print(scores)
        raise Exception("stop")
        # store the results
        self.metrics["mean_score"] = scores.mean()
        self.metrics["std_score"] = scores.std()
        self.metrics["scores"] = scores

        self._evaluated = True

        return self.metrics
    

    def report(self, write=False, output_dir=None):
        """
        Report the evaluation
        """
        if write:
            self.write_report(output_dir)
        return self.data

    def write_report(self, output_dir):
        """
        Write the evaluation report
        """
        # write to JSON
        with open(output_dir / "regression_metrics.json", "w") as f:
            json.dump(self.data, f, indent=2)
