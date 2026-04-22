import pytest

import numpy as np
from openadmet.models.eval.binary import PosthocBinaryMetrics
from openadmet.models.eval.classification import (
    ClassificationMetrics,
    ClassificationPlots,
)
from openadmet.models.eval.cross_validation import (
    PytorchLightningRepeatedKFoldCrossValidation,
    SKLearnRepeatedKFoldCrossValidation,
)
from openadmet.models.eval.eval_base import get_eval_class
from openadmet.models.eval.regression import (
    RegressionMetrics,
    RegressionPlots,
    pct_within_1_log_unit,
    relative_absolute_error,
)


def test_get_eval_class():
    get_eval_class("RegressionMetrics")
    get_eval_class("PosthocBinaryMetrics")
    get_eval_class("ClassificationMetrics")


def test_regression_metrics():
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)

    rm = RegressionMetrics()
    metrics = rm.evaluate(y_true, y_pred)

    assert metrics["task_0"]["mse"]["value"] == 0.375
    assert metrics["task_0"]["mae"]["value"] == 0.5
    assert metrics["task_0"]["r2"]["value"] == 0.9486081370449679


def test_regression_metrics_and_cv_include_rae_and_pct_within_1_log_for_pxc50():
    rm = RegressionMetrics()
    cv = SKLearnRepeatedKFoldCrossValidation()

    assert "rae" in rm.metric_names
    assert "pct_within_1_log" not in rm.metric_names
    assert "rae" in cv.metric_names
    assert "pct_within_1_log" not in cv.metric_names

    rm_pxc50 = RegressionMetrics(pXC50=True)
    cv_pxc50 = SKLearnRepeatedKFoldCrossValidation(pXC50=True)

    assert "rae" in rm_pxc50.metric_names
    assert "pct_within_1_log" in rm_pxc50.metric_names
    assert "rae" in cv_pxc50.metric_names
    assert "pct_within_1_log" in cv_pxc50.metric_names


def test_relative_absolute_error_formula():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    assert relative_absolute_error(y_true, y_pred) == pytest.approx(0.5)


def test_relative_absolute_error_denominator_zero():
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert np.isnan(relative_absolute_error(y_true, y_pred))


def test_pct_within_1_log_unit():
    y_true = np.array([6.0, 7.0, 8.0])
    y_pred = np.array([6.5, 7.2, 9.5])

    assert pct_within_1_log_unit(y_true, y_pred) == pytest.approx(2 / 3)


def test_cv_rae_scorer_is_minimization():
    cv = SKLearnRepeatedKFoldCrossValidation()
    rae_scorer, _, _ = cv._metrics["rae"]

    assert rae_scorer._sign == -1


def test_lightning_cv_pct_within_1_log_uses_raw_metric_callable():
    cv = PytorchLightningRepeatedKFoldCrossValidation(pXC50=True)
    pct_within_1_log, _, _ = cv.active_metrics["pct_within_1_log"]

    y_true = np.array([6.0, 7.0, 8.0])
    y_pred = np.array([6.5, 7.2, 9.5])

    assert pct_within_1_log(y_true, y_pred) == pytest.approx(2 / 3)


def test_regression_plots():
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)

    rm = RegressionPlots()
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


def test_posthoc_eval_metrics():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    cutoff = 4.0
    pem = PosthocBinaryMetrics()
    precision, recall = pem.get_precision_recall(y_pred, y_true, cutoff)
    assert precision == 1.0
    assert recall == 1.0
