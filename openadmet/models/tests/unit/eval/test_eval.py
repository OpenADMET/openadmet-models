import pytest

import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from sklearn.dummy import DummyRegressor
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
from openadmet.models.eval.cross_validation import (
    SKLearnRepeatedKFoldCrossValidation,
    PytorchLightningRepeatedKFoldCrossValidation,
)


def test_get_eval_class():
    get_eval_class("RegressionMetrics")
    get_eval_class("PosthocBinaryMetrics")
    get_eval_class("ClassificationMetrics")


def test_regression_metrics():
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)

    rm = RegressionMetrics(n_resamples=100)
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

    cm = ClassificationMetrics(n_resamples=100)
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


# for eval testing
@pytest.fixture
def X_all():
    return pd.DataFrame(np.random.rand(20, 5))


@pytest.fixture
def y_all():
    return pd.DataFrame(np.random.rand(20, 1))


@pytest.fixture
def pytorch_setup(tmp_path, X_all, y_all):
    fold_model = MagicMock()
    fold_model.predict.return_value = np.random.rand(10, 1)

    mock_fold_trainer = MagicMock()
    mock_fold_trainer.train.return_value = fold_model  # train() returns our fold_model

    model = MagicMock()
    model.make_new.return_value = MagicMock()

    featurizer = MagicMock()
    fold_featurizer = MagicMock()
    featurizer.make_new.return_value = fold_featurizer
    fold_featurizer.featurize.return_value = (MagicMock(), None, MagicMock(), None)

    trainer = MagicMock()
    trainer.max_epochs = 5
    trainer.accelerator = "cpu"
    trainer.devices = 1
    trainer.output_dir = tmp_path
    trainer.wandb_project = None

    return dict(
        model=model,
        fold_model=fold_model,
        mock_fold_trainer=mock_fold_trainer,
        featurizer=featurizer,
        trainer=trainer,
        X_train=X_all,
        y_train=y_all,
        X_all=X_all,
        y_all=y_all,
        tag="test",
        y_true=None,
        y_pred=None,
    )


@pytest.mark.parametrize("save_cv_models", [True, False])
@patch("openadmet.models.eval.cross_validation.LightningTrainer")
def test_pytorch_save_cv_models(mock_trainer_cls, save_cv_models, pytorch_setup):
    n_splits = 2

    mock_trainer_cls.return_value = pytorch_setup["mock_fold_trainer"]

    evaluator = PytorchLightningRepeatedKFoldCrossValidation(
        n_splits=n_splits,
        n_repeats=1,
        save_cv_models=save_cv_models,
    )

    setup = pytorch_setup
    evaluator.evaluate(
        **{
            k: v
            for k, v in setup.items()
            if k not in ("fold_model", "mock_fold_trainer")
        }
    )

    fold_model = setup["fold_model"]
    if save_cv_models:
        assert fold_model.serialize.call_count == n_splits
    else:
        fold_model.serialize.assert_not_called()


@pytest.fixture
def sklearn_setup(tmp_path, X_all, y_all):
    estimator = DummyRegressor()
    model = MagicMock()
    model.estimator = estimator

    return dict(
        model=model,
        X_train=X_all,
        y_train=y_all,
        X_all=X_all,
        y_all=y_all,
        tag="test",
        y_true=None,
        y_pred=None,
        output_dir=tmp_path,
    )


@pytest.mark.parametrize("save_cv_models", [True, False])
@patch("openadmet.models.eval.cross_validation.cross_validate")
def test_sklearn_save_cv_models(
    mock_cross_validate, save_cv_models, sklearn_setup, tmp_path
):
    n_splits = 2

    # cross_validate returns estimators only when return_estimator=True
    # mock_estimators = [MagicMock(), MagicMock()]
    mock_cross_validate.return_value = {
        "test_mse": np.array([0.1, 0.2]),
        "test_mae": np.array([0.1, 0.2]),
        "test_r2": np.array([0.8, 0.9]),
        "test_ktau": np.array([0.7, 0.8]),
        "test_spearmanr": np.array([0.7, 0.8]),
        "test_rae": np.array([0.5, 0.6]),
        "fit_time": np.array([0.1, 0.1]),
        "score_time": np.array([0.1, 0.1]),
        **(
            {"estimator": [DummyRegressor(), DummyRegressor()]}
            if save_cv_models
            else {}
        ),
    }

    evaluator = SKLearnRepeatedKFoldCrossValidation(
        n_splits=n_splits,
        n_repeats=1,
        save_cv_models=save_cv_models,
    )

    evaluator.evaluate(**sklearn_setup)

    # Verify cross_validate was called with correct return_estimator flag
    _, kwargs = mock_cross_validate.call_args
    assert kwargs["return_estimator"] == save_cv_models

    if save_cv_models:
        # Check model files were saved for each fold
        for fold in range(n_splits):
            expected_path = tmp_path / "cv" / f"fold_{fold}" / "model.pkl"
            assert expected_path.exists()
    else:
        # Check no cv model files were saved
        cv_dir = tmp_path / "cv"
        assert not cv_dir.exists()
