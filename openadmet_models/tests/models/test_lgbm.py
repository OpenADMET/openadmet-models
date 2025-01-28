import pytest
from numpy.testing import assert_allclose

from openadmet_models.models.gradient_boosting.lgbm import LGBMRegressorModel


def test_lgbm():
    lgbm_model = LGBMRegressorModel()
    assert lgbm_model.type == "LGBMRegressorModel"
    assert lgbm_model.model_params == {}


def test_lgbm_from_params():
    lgbm_model = LGBMRegressorModel.from_params(
        class_params={}, model_params={"n_estimators": 100, "boosting_type": "rf"}
    )
    assert lgbm_model.type == "LGBMRegressorModel"
    assert lgbm_model.model.get_params()["n_estimators"] == 100
    assert lgbm_model.model.get_params()["boosting_type"] == "rf"


def test_lgbm_train_predict():
    lgbm_model = LGBMRegressorModel.from_params(
        class_params={}, model_params={"n_estimators": 100}
    )
    X = [[1, 2, 3], [4, 5, 6]]
    y = [1, 2]
    lgbm_model.train(X, y)
    preds = lgbm_model.predict(X)
    assert len(preds) == 2
    assert all(preds == 1.5)

    # also test the __call__ behavior
    preds_call = lgbm_model(X)
    assert len(preds_call) == 2
    assert_allclose(preds, preds_call)
