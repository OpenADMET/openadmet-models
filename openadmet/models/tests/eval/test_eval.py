import pytest

from openadmet.models.eval.classification import (
    ClassificationMetrics,
    ClassificationPlots,
)
from openadmet.models.eval.eval_base import get_eval_class
from openadmet.models.eval.regression import RegressionMetrics


def test_get_eval_class():
    get_eval_class("RegressionMetrics")
    get_eval_class("ClassificationMetrics")


def test_regression_metrics():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]

    rm = RegressionMetrics()
    metrics = rm.evaluate(y_true, y_pred)

    assert metrics["mse"]["value"] == 0.375
    assert metrics["mae"]["value"] == 0.5
    assert metrics["r2"]["value"] == 0.9486081370449679


def test_regression_plots():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]

    rm = RegressionMetrics()
    rm.evaluate(y_true, y_pred)

    assert True


def test_classification_metrics():
    y_true = [0, 1, 1, 0]

    # We pass probabilities of the class, not the class itself
    # Classes would be [0, 1, 1, 1]
    y_pred = [[1, 0], [0, 1], [0, 1], [0, 1]]

    cm = ClassificationMetrics()
    metrics = cm.evaluate(y_true, y_pred)

    assert metrics["accuracy"]["value"] == 0.75
    assert metrics["precision"]["value"] == pytest.approx(0.667, abs=0.001)
    assert metrics["recall"]["value"] == 1.0
    assert metrics["f1"]["value"] == 0.8
    assert metrics["roc_auc"]["value"] == 0.75
    assert metrics["pr_auc"]["value"] == pytest.approx(0.833, abs=0.001)


def test_classification_plots():
    y_true = [0, 1, 1, 0]
    y_pred = [[1, 0], [0, 1], [0, 1], [0, 1]]

    cp = ClassificationPlots()
    cp.evaluate(y_true, y_pred)

    assert True
