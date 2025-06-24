import pytest

from openadmet.models.architecture.model_base import (
    PickleableModelBase,
    models,
)


@pytest.mark.parametrize("mclass", models.values())
def test_save_load_pickleable(mclass, tmp_path):
    if not issubclass(mclass, PickleableModelBase):
        pytest.skip(f"Skipping non-pickleable model {mclass.__name__}")
    model = mclass()
    model.build()
    model.save(tmp_path / "test_model.pkl")
    loaded_model = mclass()
    loaded_model.build()
    loaded_model.load(tmp_path / "test_model.pkl")
