import pytest
from numpy.testing import assert_allclose

from openadmet.models.architecture.lgbm import LGBMRegressorModel


@pytest.fixture
def X_y():
    """Provide simple synthetic data for basic model training tests."""
    X = [[1, 2, 3], [4, 5, 6]]
    y = [1, 2]
    return X, y


def test_lgbm():
    """Verify that LGBMRegressorModel initializes with the correct type identifier."""
    lgbm_model = LGBMRegressorModel()
    assert lgbm_model.type == "LGBMRegressorModel"


def test_lgbm_from_params():
    """
    Validate that hyperparameters passed to the constructor are correctly applied to the underlying estimator.

    This ensures that user configurations (like n_estimators) are respected by the model.
    """
    lgbm_model = LGBMRegressorModel(n_estimators=100, boosting_type="rf")
    lgbm_model.build()
    assert lgbm_model.type == "LGBMRegressorModel"
    assert lgbm_model.estimator.get_params()["n_estimators"] == 100
    assert lgbm_model.estimator.get_params()["boosting_type"] == "rf"


def test_lgbm_train_predict(X_y):
    """
    Verify the train and predict lifecycle of LGBMRegressorModel.

    This checks that the model can fit to data and generate predictions with the expected shape and values.
    """
    lgbm_model = LGBMRegressorModel(n_estimators=100)
    lgbm_model.build()
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
    """
    Validate persistence of the LGBM model to disk.

    Ensures that saving and reloading the model preserves its learned state and prediction behavior.
    """
    lgbm_model = LGBMRegressorModel(n_estimators=100)
    lgbm_model.build()
    X, y = X_y
    lgbm_model.train(X, y)
    save_path = tmp_path / "lgbm_model.pkl"
    preds = lgbm_model.predict(X)
    lgbm_model.save(save_path)
    lgbm_model.load(save_path)
    preds2 = lgbm_model.predict(X)
    assert_allclose(preds, preds2)


def test_serialization(tmp_path, X_y):
    """
    Validate JSON/pickle serialization workflow for LGBM models.

    This tests the separate storage of hyperparameters (JSON) and model weights (pickle),
    which is used for model registry and versioning.
    """
    lgbm_model = LGBMRegressorModel(n_estimators=100)
    lgbm_model.build()
    X, y = X_y
    lgbm_model.train(X, y)
    preds = lgbm_model.predict(X)
    lgbm_model.serialize(tmp_path / "lgbm_model.json", tmp_path / "lgbm_model.pkl")
    lgbm_model_loaded = LGBMRegressorModel.deserialize(
        tmp_path / "lgbm_model.json", tmp_path / "lgbm_model.pkl"
    )
    preds_loaded = lgbm_model_loaded.predict(X)
    assert_allclose(preds, preds_loaded)
