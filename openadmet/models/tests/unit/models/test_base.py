import pytest
import torch
from unittest.mock import Mock

import openadmet.models.architecture.lightning_model_base as lightning_model_base
from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.architecture.model_base import (
    LightningModelBase,
    PickleableModelBase,
    models,
)
from openadmet.models.architecture.tabicl import TabICLModelBase
from openadmet.models.architecture.tabpfn import TabPFNModelBase

# TabPFN and TabICL save()/load() persist a fitted estimator's state, not a
# plain pickle of self.estimator, so they cannot round-trip right after
# build() (before train()). Their own save/load round-trip tests in
# test_tabpfn.py / test_tabicl.py cover the fitted-state contract instead.
_REQUIRES_FIT_BEFORE_SAVE = (TabPFNModelBase, TabICLModelBase)


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
    if issubclass(mclass, _REQUIRES_FIT_BEFORE_SAVE):
        pytest.skip(
            f"{mclass.__name__} requires a fitted estimator before save(); "
            "covered by its own save/load round-trip tests"
        )
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
