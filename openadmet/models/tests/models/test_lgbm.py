import pytest
from numpy.testing import assert_allclose

from openadmet.models.architecture.lgbm import LGBMRegressorModel


@pytest.fixture
def X_y():
    X = [[1, 2, 3], [4, 5, 6]]
    y = [1, 2]
    return X, y


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


def test_lgbm_train_predict(X_y):
    lgbm_model = LGBMRegressorModel.from_params(
        class_params={}, model_params={"n_estimators": 100}
    )
    X, y = X_y

    lgbm_model.train(X, y)
    preds = lgbm_model.predict(X)
    assert len(preds) == 2
    assert all(preds == 1.5)

    # also test the __call__ behavior
    preds_call = lgbm_model(X)
    assert len(preds_call) == 2
    assert_allclose(preds, preds_call)


def test_lgbm_save_load(tmp_path, X_y):
    lgbm_model = LGBMRegressorModel.from_params(
        class_params={}, model_params={"n_estimators": 100}
    )
    X, y = X_y
    lgbm_model.train(X, y)
    save_path = tmp_path / "lgbm_model.pkl"
    preds = lgbm_model.predict(X)
    lgbm_model.save(save_path)
    lgbm_model.load(save_path)
    preds2 = lgbm_model.predict(X)
    assert_allclose(preds, preds2)


def test_serialization(tmp_path, X_y):
    lgbm_model = LGBMRegressorModel.from_params(
        class_params={}, model_params={"n_estimators": 100}
    )
    X, y = X_y
    lgbm_model.train(X, y)
    preds = lgbm_model.predict(X)
    lgbm_model.serialize(tmp_path / "lgbm_model.json", tmp_path / "lgbm_model.pkl")
    lgbm_model_loaded = LGBMRegressorModel.deserialize(
        tmp_path / "lgbm_model.json", tmp_path / "lgbm_model.pkl"
    )
    preds_loaded = lgbm_model_loaded.predict(X)
    assert_allclose(preds, preds_loaded)
