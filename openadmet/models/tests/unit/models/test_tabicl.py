"""Unit tests for TabICL models."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_regressor_build_train_predict():
    """build must construct the TabICL estimator, and train plus predict must produce 2D predictions."""
    from tabicl import TabICLRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)

    model = TabICLRegressorModel(n_estimators=1, accelerator="cpu")
    model.build()
    assert isinstance(model.estimator, TabICLRegressor)

    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (20, 1)
    assert np.isfinite(preds).all()


def test_predict_raises_if_not_trained():
    """Predict should raise when model is not built."""
    model = TabICLRegressorModel()
    with pytest.raises(ValueError):
        model.predict(np.zeros((1, 2)))


def test_predict_accepts_pipeline_kwargs():
    """predict must accept extra kwargs such as accelerator and ignore them, since the anvil inference path passes them."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)

    model = TabICLRegressorModel(n_estimators=1, accelerator="cpu")
    model.train(X, y)

    out_plain = model.predict(X)
    out_pipelined = model.predict(X, accelerator="cpu")

    assert out_plain.shape == (20, 1)
    np.testing.assert_allclose(out_plain, out_pipelined, rtol=1e-12)


def test_classifier_predict_proba():
    """Classifier proba must return one row of class probabilities per sample."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = np.array([0, 1] * 10)

    model = TabICLClassifierModel(n_estimators=1, accelerator="cpu")
    model.train(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (20, 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_registry_names():
    """Ensure models are registered with correct keys."""
    from openadmet.models.architecture.model_base import models

    assert "TabICLRegressorModel" in models._registry
    assert "TabICLClassifierModel" in models._registry
    assert models.get_class("TabICLRegressorModel") is TabICLRegressorModel
    assert models.get_class("TabICLClassifierModel") is TabICLClassifierModel
