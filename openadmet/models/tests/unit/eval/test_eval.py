import matplotlib.figure
import numpy as np
import seaborn as sns

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
    """Verify that evaluation classes can be retrieved by name from the registry."""
    get_eval_class("RegressionMetrics")
    get_eval_class("PosthocBinaryMetrics")
    get_eval_class("ClassificationMetrics")


def test_regression_metrics():
    """
    Validate calculation of standard regression metrics (MSE, MAE, R2).

    This test uses simple synthetic data to ensure that the mathematical implementations
    of these metrics are correct and return the expected values.
    """
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)

    rm = RegressionMetrics(n_resamples=100)
    metrics = rm.evaluate(y_true, y_pred)

    assert np.allclose(metrics["task_0"]["mse"]["value"], 0.375, atol=0.001)
    assert np.allclose(metrics["task_0"]["mae"]["value"], 0.5, atol=0.001)
    assert np.allclose(metrics["task_0"]["r2"]["value"], 0.94860, atol=0.001)


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

    assert np.allclose(relative_absolute_error(y_true, y_pred), 0.5)


def test_relative_absolute_error_denominator_zero():
    y_true = np.array([2.0, 2.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert np.isnan(relative_absolute_error(y_true, y_pred))


def test_pct_within_1_log_unit():
    y_true = np.array([6.0, 7.0, 8.0])
    y_pred = np.array([6.5, 7.2, 9.5])

    assert np.allclose(pct_within_1_log_unit(y_true, y_pred), 2 / 3)


def test_cv_rae_scorer_is_minimization():
    cv = SKLearnRepeatedKFoldCrossValidation()
    rae_scorer, _, _ = cv._metrics["rae"]

    assert rae_scorer._sign == -1


def test_lightning_cv_pct_within_1_log_uses_raw_metric_callable():
    cv = PytorchLightningRepeatedKFoldCrossValidation(pXC50=True)
    pct_within_1_log, _, _ = cv.active_metrics["pct_within_1_log"]

    y_true = np.array([6.0, 7.0, 8.0])
    y_pred = np.array([6.5, 7.2, 9.5])

    assert np.allclose(pct_within_1_log(y_true, y_pred), 2 / 3)


def test_regression_plots():
    """
    Verify that regression plotting functions return valid figure objects.

    This ensures that regression plots (JointGrid for parity, Figure for CI) are generated
    without error, which is important for model reporting.
    """
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)

    rm = RegressionPlots()
    plot_data = rm.evaluate(y_true, y_pred)

    assert isinstance(plot_data, dict)
    assert "task_0_regplot" in plot_data
    assert "task_0_ciplot" in plot_data
    assert isinstance(plot_data["task_0_regplot"], sns.axisgrid.JointGrid)
    assert isinstance(plot_data["task_0_ciplot"], matplotlib.figure.Figure)


def test_classification_metrics():
    """
    Validate calculation of classification metrics (Accuracy, Precision, Recall, F1, AUC).

    This ensures that for binary classification tasks, the metrics are computed correctly based on
    predicted probabilities and ground truth labels.
    """
    y_true = [0, 1, 1, 0]

    # We pass probabilities of the class, not the class itself
    # Classes would be [0, 1, 1, 1]
    y_pred = [[1, 0], [0, 1], [0, 1], [0, 1]]

    cm = ClassificationMetrics(n_resamples=100)
    metrics = cm.evaluate(y_true, y_pred)

    assert np.allclose(metrics["accuracy"]["value"], 0.75)
    assert np.allclose(metrics["precision"]["value"], 0.667, atol=0.001)
    assert np.allclose(metrics["recall"]["value"], 1.0)
    assert np.allclose(metrics["f1"]["value"], 0.8)
    assert np.allclose(metrics["roc_auc"]["value"], 0.75)
    assert np.allclose(metrics["pr_auc"]["value"], 0.833, atol=0.001)


def test_classification_plots():
    """
    Verify that classification plotting functions (ROC, PR curves) return valid figure objects.
    """
    y_true = [0, 1, 1, 0]
    y_pred = [[1, 0], [0, 1], [0, 1], [0, 1]]

    cp = ClassificationPlots()
    cp.evaluate(y_true, y_pred)

    assert isinstance(cp.plot_data, dict)
    assert "roc_curve" in cp.plot_data
    assert "pr_curve" in cp.plot_data
    assert isinstance(cp.plot_data["roc_curve"], matplotlib.figure.Figure)
    assert isinstance(cp.plot_data["pr_curve"], matplotlib.figure.Figure)


def test_posthoc_eval_metrics():
    """
    Test post-hoc binary metrics utility functions.

    Verifies that we can calculate precision and recall at a specific cutoff threshold from
    regression-like outputs (or probabilities).
    """
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    cutoff = 4.0
    pem = PosthocBinaryMetrics()
    precision, recall = pem.get_precision_recall(y_pred, y_true, cutoff)
    assert precision == 1.0
    assert recall == 1.0
