"""Unit tests for TabPFN models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openadmet.models.architecture.tabpfn import (
    TabPFNClassifierModel,
    TabPFNModelBase,
    TabPFNPostHocClassifierModel,
    TabPFNPostHocRegressorModel,
    TabPFNRegressorModel,
    _resolve_device,
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


class TestTabPFNModelBase:
    """Tests for TabPFNModelBase (basic tabpfn)."""

    def test_default_fields(self):
        """Verify default field values."""
        model = TabPFNModelBase()
        assert model.random_seed == 42
        assert model.accelerator == "auto"
        assert model.ignore_pretraining_limits is False

    def test_accelerator_validator_rejects_bad_value(self):
        """An accelerator torch.device() cannot parse must fail at construction time."""
        with pytest.raises(ValueError, match="Invalid accelerator"):
            TabPFNModelBase(accelerator="not_a_real_device")

    @pytest.mark.parametrize("accelerator", ["cpu", "gpu", "auto", "mps", "cuda:0"])
    def test_accelerator_validator_accepts_known_values(self, accelerator):
        """cpu, gpu, auto, and unaliased torch device spellings must all construct cleanly."""
        TabPFNModelBase(accelerator=accelerator)

    def test_predict_raises_if_not_trained(self):
        """Predict should raise when model is not built."""
        model = TabPFNModelBase()
        with pytest.raises(ValueError):
            model.predict(np.zeros((1, 2)))


class TestTabPFNExtensionModelBase:
    """Tests for TabPFNExtensionModelBase (tabpfn-extensions)."""

    def test_default_fields(self):
        """Verify default field values."""
        model = TabPFNPostHocRegressorModel()
        assert model.random_seed == 42
        assert model.accelerator == "auto"
        assert model.max_time is None
        assert model.ignore_pretraining_limits is False
        assert model.phe_init_args is None

    def test_accelerator_validator_rejects_bad_value(self):
        with pytest.raises(ValueError, match="Invalid accelerator"):
            TabPFNPostHocRegressorModel(accelerator="bogus")

    @pytest.mark.parametrize("accelerator", ["cpu", "gpu", "auto", "mps", "cuda:0"])
    def test_accelerator_validator_accepts_known_values(self, accelerator):
        TabPFNPostHocRegressorModel(accelerator=accelerator)

    @pytest.mark.parametrize(
        "model_cls",
        [TabPFNPostHocRegressorModel, TabPFNPostHocClassifierModel],
    )
    def test_predict_raises_if_not_trained(self, model_cls):
        """Predict should raise when model is not built."""
        model = model_cls()
        with pytest.raises(ValueError):
            model.predict(np.zeros((1, 2)))

    def test_registry_names(self):
        """Ensure extension models are registered with correct keys."""
        from openadmet.models.architecture.model_base import models

        assert "TabPFNPostHocRegressorModel" in models._registry
        assert "TabPFNPostHocClassifierModel" in models._registry
        assert (
            models.get_class("TabPFNPostHocRegressorModel")
            is TabPFNPostHocRegressorModel
        )
        assert (
            models.get_class("TabPFNPostHocClassifierModel")
            is TabPFNPostHocClassifierModel
        )

    def test_build_license_warning(self):
        """build() must emit the TabPFN license warning."""
        model = TabPFNPostHocRegressorModel(accelerator="cpu")
        with pytest.warns(UserWarning, match="TabPFN 2.5 License"):
            try:
                model.build()
            except (ImportError, ModuleNotFoundError):
                pass  # tabpfn-extensions not installed — warning should fire before import


class TestTabPFNBasicModels:
    """Tests for TabPFNRegressorModel and TabPFNClassifierModel (basic tabpfn)."""

    @pytest.mark.parametrize(
        "model_cls",
        [TabPFNRegressorModel, TabPFNClassifierModel],
    )
    def test_predict_raises_if_not_trained(self, model_cls):
        model = model_cls()
        with pytest.raises(ValueError):
            model.predict(np.zeros((1, 2)))

    def test_registry_names(self):
        """Ensure basic models are registered with correct keys."""
        from openadmet.models.architecture.model_base import models

        assert "TabPFNRegressorModel" in models._registry
        assert "TabPFNClassifierModel" in models._registry
        assert models.get_class("TabPFNRegressorModel") is TabPFNRegressorModel
        assert models.get_class("TabPFNClassifierModel") is TabPFNClassifierModel


class TestResolveDevice:
    """Tests for the private _resolve_device helper."""

    @pytest.mark.parametrize(
        "accelerator,expected",
        [
            ("auto", "auto"),
            ("gpu", "cuda"),
            ("tpu", "xla"),
            ("cpu", "cpu"),
            ("cuda:0", "cuda:0"),
            ("mps", "mps"),
        ],
    )
    def test_resolve_device(self, accelerator, expected):
        assert _resolve_device(accelerator) == expected
