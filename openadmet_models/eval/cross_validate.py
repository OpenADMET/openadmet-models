from sklearn.model_selection import KFold, cross_validate, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from openadmet_models.eval.eval_base import EvalBase, evaluators
from openadmet_models.eval.regression import stat_and_bootstrap, nan_omit_ktau, nan_omit_spearmanr
from scipy.stats import bootstrap, kendalltau, spearmanr
from sklearn.metrics import make_scorer
import json
from loguru import logger

def wrap_ktau(y_true, y_pred):
    return nan_omit_ktau(y_true, y_pred).statistic

def wrap_spearmanr(y_true, y_pred):
    return nan_omit_spearmanr(y_true, y_pred).correlation

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

        # store the metric names and callables in dict suitable for sklearn cross_validate
        self.metrics = {
            "mse": make_scorer(mean_squared_error),
            "mae": make_scorer(mean_absolute_error),
            "r2": make_scorer(r2_score),
            "ktau": make_scorer(wrap_ktau),
            "spearmanr": make_scorer(wrap_spearmanr),
        }

        logger.info("Starting cross-validation")
        
        # run CV
        cv = RepeatedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
        )

        estimator = model.model
        # evaluate the model,
        scores = cross_validate(estimator, X_train, y_train, cv=cv, n_jobs=-1, scoring = self.metrics)

        logger.info("Cross-validation complete")
        
        # remove the 'test_' prefix from the keys
        # also convert the numpy arrays to lists so they can be serialized to JSON
        clean_scores = {}
        for k, v in scores.items():
            clean_scores[k.replace("test_", "")] = v.tolist()

        self.data = clean_scores
        self._evaluated = True

        return self.data
    

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
        with open(output_dir / "cross_validation_metrics.json", "w") as f:
            json.dump(self.data, f, indent=2)
