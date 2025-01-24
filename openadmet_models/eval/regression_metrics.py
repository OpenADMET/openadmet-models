from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from openadmet_models.eval.eval_base import EvalBase, EVAL_CLASSES


@EVAL_CLASSES.register("RegressionMetrics")
class RegressionMetrics(EvalBase):

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the regression model
        """

        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

        return metrics
        