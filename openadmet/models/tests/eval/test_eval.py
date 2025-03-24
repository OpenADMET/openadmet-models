import pytest

from openadmet.models.eval.eval_base import get_eval_class
from openadmet.models.eval.regression import RegressionMetrics


def test_get_eval_class():
    get_eval_class("RegressionMetrics")
    with pytest.raises(ValueError):
        get_eval_class("ClassificationMetrics")


def test_regression_metrics():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    rm = RegressionMetrics()
    metrics = rm.evaluate(y_true, y_pred)
    assert metrics["mse"]["value"] == 0.375
    assert metrics["mae"]["value"] == 0.5
    assert metrics["r2"]["value"] == 0.9486081370449679
