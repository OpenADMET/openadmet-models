"""Unit tests for TabICL models."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from openadmet.models.architecture.tabicl import (
    TabICLClassifierModel,
    TabICLModelBase,
    TabICLRegressorModel,
)


def test_tabicl_model_base_fields():
    """Verify default fields and mapping."""
    model = TabICLModelBase()
    assert model.random_seed == 42
    assert model.accelerator == "auto"
    assert model.n_estimators == 8
    assert model.batch_size == 1
    assert model.use_amp == "auto"
    assert model.use_fa3 == "auto"
    assert model.offload_mode == "auto"


def test_accelerator_validator():
    """Validate accelerator values."""
    with pytest.raises(ValueError):
        TabICLModelBase(accelerator="invalid")


def test_build_kwargs_mapping():
    """Ensure public names map to estimator names."""
    model = TabICLRegressorModel(
        random_seed=123,
        accelerator="gpu",
        n_estimators=4,
        batch_size=2,
        use_amp="no",
        use_fa3="yes",
        offload_mode="disk",
        norm_methods=["standard"],
    )
    kwargs = model._build_kwargs()
    assert kwargs["random_state"] == 123
    assert kwargs["device"] == "cuda"
    assert kwargs["n_estimators"] == 4
    assert kwargs["batch_size"] == 2
    assert kwargs["use_amp"] == "no"
    assert kwargs["use_fa3"] == "yes"
    assert kwargs["offload_mode"] == "disk"
    assert kwargs["norm_methods"] == ["standard"]
    # Unknown extra fields should be ignored
    model2 = TabICLRegressorModel(extra_param="keep me")
    kwargs2 = model2._build_kwargs()
    assert "extra_param" not in kwargs2


@patch("tabicl.TabICLRegressor")
def test_regressor_build_and_train(mock_est_cls):
    """Test build and train flow with mocked estimator."""
    mock_est = MagicMock()
    mock_est.fit.return_value = mock_est
    mock_est_cls.return_value = mock_est

    model = TabICLRegressorModel(random_seed=1, accelerator="cpu")
    model.build()
    assert model.estimator is mock_est

    X = np.zeros((5, 3))
    y = np.zeros(5)
    model.train(X, y)
    mock_est.fit.assert_called_once_with(X, y)


@patch("tabicl.TabICLRegressor")
def test_regressor_predict_shape(mock_est_cls):
    """Test predict returns correct shape."""
    mock_est = MagicMock()
    mock_est.predict.return_value = np.array([0.1, 0.2, 0.3])
    mock_est_cls.return_value = mock_est

    model = TabICLRegressorModel()
    model.build()
    X = np.zeros((3, 2))
    preds = model.predict(X)
    assert preds.shape == (3, 1)
    np.testing.assert_array_equal(preds.ravel(), [0.1, 0.2, 0.3])


def test_predict_raises_if_not_trained():
    """Predict should raise when model is not built."""
    model = TabICLRegressorModel()
    with pytest.raises(ValueError):
        model.predict(np.zeros((1, 2)))


def test_predict_rejects_kwargs():
    """Predict should reject unknown kwargs."""
    from unittest.mock import MagicMock

    mock_est = MagicMock()
    mock_est.predict.return_value = np.array([0.1])
    model = TabICLRegressorModel()
    model.estimator = mock_est
    with pytest.raises(TypeError):
        model.predict(np.zeros((1, 2)), unknown=1)


@patch("tabicl.TabICLClassifier")
def test_classifier_predict_proba(mock_est_cls):
    """Test classifier predict_proba delegation."""
    mock_est = MagicMock()
    mock_est.predict_proba.return_value = np.array([[0.2, 0.8]])
    mock_est_cls.return_value = mock_est

    model = TabICLClassifierModel()
    model.build()
    X = np.zeros((1, 2))
    proba = model.predict_proba(X)
    np.testing.assert_array_equal(proba, [[0.2, 0.8]])


def test_registry_names():
    """Ensure models are registered with correct keys."""
    from openadmet.models.architecture.model_base import models

    assert "TabICLRegressorModel" in models._registry
    assert "TabICLClassifierModel" in models._registry
    assert models.get_class("TabICLRegressorModel") is TabICLRegressorModel
    assert models.get_class("TabICLClassifierModel") is TabICLClassifierModel
