"""Unit tests for TabICL models."""

from __future__ import annotations

import numpy as np
import pytest

from openadmet.models.architecture.tabicl import (
    TabICLClassifierModel,
    TabICLModelBase,
    TabICLRegressorModel,
)


@pytest.fixture
def regression_data():
    """20-sample, 4-feature regression data for train/predict tests."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)
    return X, y


@pytest.fixture
def classification_data():
    """20-sample, 4-feature, 2-class classification data for train/predict tests."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = np.array([0, 1] * 10)
    return X, y


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


def test_accelerator_validator_rejects_bad_value():
    """An accelerator torch.device() cannot parse must fail at construction time."""
    with pytest.raises(ValueError, match="Invalid accelerator"):
        TabICLModelBase(accelerator="not_a_real_device")


@pytest.mark.parametrize("accelerator", ["cpu", "gpu", "auto", "mps", "cuda:0"])
def test_accelerator_validator_accepts_known_values(accelerator):
    """cpu, gpu, auto, and unaliased torch device spellings must all construct cleanly."""
    TabICLModelBase(accelerator=accelerator)


def test_build_kwargs_mapping():
    """Ensure public names map to estimator names."""
    model = TabICLRegressorModel(
        random_seed=123,
        accelerator="cpu",
        n_estimators=4,
        batch_size=2,
        use_amp="no",
        use_fa3="yes",
        offload_mode="disk",
        norm_methods=["standard"],
    )
    kwargs = model._build_kwargs()
    assert kwargs["random_state"] == 123
    assert kwargs["n_estimators"] == 4
    assert kwargs["batch_size"] == 2
    assert kwargs["use_amp"] == "no"
    assert kwargs["use_fa3"] == "yes"
    assert kwargs["offload_mode"] == "disk"
    assert kwargs["norm_methods"] == ["standard"]


@pytest.mark.parametrize(
    "accelerator,expected_device",
    [
        ("gpu", "cuda"),
        ("auto", None),
        ("tpu", "xla"),
        ("mps", "mps"),
        ("cuda:0", "cuda:0"),
    ],
)
def test_build_kwargs_maps_accelerator_to_device(accelerator, expected_device):
    """Accelerator aliases resolve to torch device names; "auto" maps to None so
    TabICL runs its own device detection, and unaliased spellings pass through."""
    model = TabICLRegressorModel(accelerator=accelerator)
    kwargs = model._build_kwargs()
    assert kwargs["device"] == expected_device


def test_build_raises_on_unsupported_kwarg():
    """Unsupported extra fields must fail loudly at build time, not be dropped."""
    model = TabICLRegressorModel(not_a_real_param="keep me")
    with pytest.raises(TypeError):
        model.build()


def test_regressor_build_train_predict(regression_data):
    """build must construct the TabICL estimator, and train plus predict must produce 2D predictions."""
    from tabicl import TabICLRegressor

    X, y = regression_data
    model = TabICLRegressorModel(n_estimators=1, accelerator="cpu")
    model.build()
    assert isinstance(model.estimator, TabICLRegressor)

    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (20, 1)
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("model_cls", [TabICLRegressorModel, TabICLClassifierModel])
def test_predict_raises_if_not_trained(model_cls):
    """Predict should raise when model is not built."""
    model = model_cls()
    with pytest.raises(ValueError):
        model.predict(np.zeros((1, 2)))


def test_predict_accepts_pipeline_kwargs(regression_data):
    """predict must accept extra kwargs such as accelerator and ignore them, since the anvil inference path passes them."""
    X, y = regression_data
    model = TabICLRegressorModel(n_estimators=1, accelerator="cpu")
    model.train(X, y)

    out_plain = model.predict(X)
    out_pipelined = model.predict(X, accelerator="cpu")

    assert out_plain.shape == (20, 1)
    np.testing.assert_allclose(out_plain, out_pipelined, rtol=1e-12)


def test_classifier_predict_proba(classification_data):
    """Classifier proba must return one row of class probabilities per sample."""
    X, y = classification_data
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
