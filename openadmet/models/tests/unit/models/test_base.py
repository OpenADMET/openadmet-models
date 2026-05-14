from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import openadmet.models.architecture.model_base as model_base
from openadmet.models.architecture.model_base import (
    LightningModelBase,
    PickleableModelBase,
    models,
)


@pytest.mark.parametrize("mclass", models.classes())
def test_save_load_pickleable(mclass, tmp_path):
    """
    Verify save/load mechanics for all registered pickleable models (e.g., sklearn-based).

    This iterates through the model registry and tests that any model inheriting from PickleableModelBase
    can be instantiated, built, saved, and loaded without error. This is a crucial contract test
    ensuring all registered models comply with the persistence interface.
    """
    if not issubclass(mclass, PickleableModelBase):
        pytest.skip(f"Skipping non-pickleable model {mclass.__name__}")
    model = mclass()
    model.build()
    model.save(tmp_path / "test_model.pkl")
    loaded_model = mclass()
    loaded_model.build()
    loaded_model.load(tmp_path / "test_model.pkl")


@pytest.mark.parametrize("mclass", models.classes())
def test_save_load_torch_model(mclass, tmp_path):
    """
    Verify save/load mechanics for all registered PyTorch Lightning models.

    Similar to the pickleable test, this ensures that deep learning models (inheriting from LightningModelBase)
    implement the correct save/load logic for their weights and configurations.
    """
    if not issubclass(mclass, LightningModelBase):
        pytest.skip(f"Skipping non-torch model {mclass.__name__}")
    model = mclass()
    model.build()
    model.save(tmp_path / "test_model.pth")
    loaded_model = mclass()
    loaded_model.build()
    loaded_model.load(tmp_path / "test_model.pth")


def test_lightning_model_load_uses_weights_only(monkeypatch, tmp_path):
    state_dict = {"layer.weight": "dummy"}
    torch_load = Mock(return_value=state_dict)
    monkeypatch.setattr(model_base.torch, "load", torch_load)

    estimator = Mock()
    model = SimpleNamespace(estimator=estimator)
    path = tmp_path / "test_model.pth"

    LightningModelBase.load(model, path)

    torch_load.assert_called_once_with(path, weights_only=True)
    estimator.load_state_dict.assert_called_once_with(state_dict)
