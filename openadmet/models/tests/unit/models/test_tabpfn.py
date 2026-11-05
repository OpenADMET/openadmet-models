"""Unit tests for TabPFN models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openadmet.models.architecture.tabpfn import (
    TabPFNClassifierModel,
    TabPFNModelBase,
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


class _FakeFittedEstimator:
    """
    Stand-in for a fitted TabPFN estimator.

    Mirrors the real ``save_fit_state``/``load_from_fit_state`` contract
    (tabpfn.TabPFNRegressor/TabPFNClassifier) without downloading license-gated
    model weights over the network, which the real estimator requires on
    first fit.
    """

    def __init__(self, value: float):
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return `value` for every row, so a round trip is checkable."""
        return np.full(len(X), self.value)

    def save_fit_state(self, path) -> None:
        """Write `value` to `path`, mirroring the real instance-method API."""
        with open(path, "w") as f:
            f.write(str(self.value))

    @classmethod
    def load_from_fit_state(cls, path, *, device: str = "cpu"):
        """Read `value` back from `path`, mirroring the real classmethod API."""
        with open(path) as f:
            return cls(float(f.read()))


class TestTabPFNSaveLoad:
    """Tests for TabPFNModelBase.save/load against the real fit-state API shape."""

    @pytest.mark.parametrize(
        "model_cls",
        [TabPFNRegressorModel, TabPFNClassifierModel],
    )
    def test_save_load_round_trip(self, monkeypatch, tmp_path, model_cls):
        """save() must call estimator.save_fit_state and load() must call the class's load_from_fit_state."""
        monkeypatch.setattr(
            model_cls,
            "_get_estimator_class",
            classmethod(lambda cls: _FakeFittedEstimator),
        )
        model = model_cls(accelerator="cpu")
        model.estimator = _FakeFittedEstimator(3.5)
        path = tmp_path / "model.tabpfn_fit"

        model.save(path)
        reloaded = model_cls(accelerator="cpu")
        reloaded.load(path)

        assert reloaded.predict(np.zeros((2, 4))).flatten().tolist() == [3.5, 3.5]

    @pytest.mark.parametrize(
        "model_cls",
        [TabPFNRegressorModel, TabPFNClassifierModel],
    )
    def test_save_raises_if_not_trained(self, model_cls):
        """save() must raise rather than silently writing an empty/invalid file."""
        model = model_cls()
        with pytest.raises(ValueError):
            model.save("unused.tabpfn_fit")


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
