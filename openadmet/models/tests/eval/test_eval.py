import pytest

from openadmet.models.eval.eval_base import get_eval_class
from openadmet.models.eval.regression import RegressionMetrics
from openadmet.models.eval.binary import PosthocBinaryMetrics


def test_get_eval_class():
    get_eval_class("RegressionMetrics")
    get_eval_class("PosthocBinaryMetrics")
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

def test_posthoc_eval_metrics():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    cutoff = 4.
    pem = PosthocBinaryMetrics()
    precision, recall = pem.get_precision_recall(y_pred, y_true, cutoff)
    assert precision == 1.0
    assert recall == 1.0
