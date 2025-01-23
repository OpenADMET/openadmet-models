import pytest
from openadmet_models.models.model_catalouge import MODEL_CLASSES
from openadmet_models.models.gradient_boosting.lgbm import LGBMRegressorModel



def test_model_catalouge():
    assert len(MODEL_CLASSES) > 0

@pytest.mark.parametrize("mclass", MODEL_CLASSES.values())
def test_save_load_pickleable(mclass, tmp_path):
    model = mclass()
    model.build()
    model.save(tmp_path / "test_model.pkl")
    loaded_model =  mclass()
    loaded_model.load(tmp_path / "test_model.pkl")

