"""Unit tests for linear models with mean imputation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import GridSearchCV

from openadmet.models.architecture.linear import (
    ElasticNetModel,
    LassoModel,
    LogisticRegressionL1Model,
    LogisticRegressionL2Model,
    RidgeModel,
)


@pytest.fixture
def regression_data():
    """Create simple regression data for testing."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    return X, y


@pytest.fixture
def regression_data_with_nans():
    """Create regression data with NaN values for imputation testing."""
    X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, np.nan], [10, 11, 12]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    return X, y


@pytest.fixture
def classification_data():
    """Create simple classification data for testing."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    return X, y


@pytest.fixture
def classification_data_with_nans():
    """Create classification data with NaN values for imputation testing."""
    X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, np.nan], [10, 11, 12]])
    y = np.array([0, 1, 0, 1])
    return X, y


def test_ridge_model_basic(regression_data):
    """Test Ridge model basic functionality."""
    model = RidgeModel(alpha=1.0)
    X, y = regression_data
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (4, 1)


def test_ridge_model_with_mean_imputation(regression_data_with_nans):
    """Test Ridge model with mean imputation."""
    model = RidgeModel(alpha=1.0, use_mean_imputation=True)
    X, y = regression_data_with_nans
    model.train(X, y)

    # Verify imputer was created and fitted
    assert model._imputer is not None
    # Check that imputer learned correct means: col 1 mean = (2+8+11)/3 = 7
    # col 2 mean = (3+6+12)/3 = 7
    expected_means = np.array([5.5, 7.0, 7.0])
    assert_allclose(model._imputer.statistics_, expected_means)

    # Test prediction with NaN data
    preds = model.predict(X)
    assert preds.shape == (4, 1)
    assert not np.any(np.isnan(preds))

    # Test prediction with new NaN data
    X_new = np.array([[np.nan, np.nan, 5], [2, 3, np.nan]])
    preds_new = model.predict(X_new)
    assert preds_new.shape == (2, 1)
    assert not np.any(np.isnan(preds_new))


def test_lasso_model_with_mean_imputation(regression_data_with_nans):
    """Test Lasso model with mean imputation."""
    model = LassoModel(alpha=0.1, use_mean_imputation=True)
    X, y = regression_data_with_nans
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (4, 1)
    assert not np.any(np.isnan(preds))


def test_elasticnet_model_with_mean_imputation(
    regression_data_with_nans,
):
    """Test ElasticNet model with mean imputation."""
    model = ElasticNetModel(alpha=0.1, l1_ratio=0.5, use_mean_imputation=True)
    X, y = regression_data_with_nans
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (4, 1)
    assert not np.any(np.isnan(preds))


def test_logistic_regression_l1_with_mean_imputation(
    classification_data_with_nans,
):
    """Test LogisticRegression L1 model with mean imputation."""
    model = LogisticRegressionL1Model(C=1.0, use_mean_imputation=True, random_state=42)
    X, y = classification_data_with_nans
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (4, 1)
    assert not np.any(np.isnan(preds))


def test_logistic_regression_l2_with_mean_imputation(
    classification_data_with_nans,
):
    """Test LogisticRegression L2 model with mean imputation."""
    model = LogisticRegressionL2Model(C=1.0, use_mean_imputation=True, random_state=42)
    X, y = classification_data_with_nans
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (4, 1)
    assert not np.any(np.isnan(preds))


def test_ridge_model_imputation_disabled_fails_with_nans(
    regression_data_with_nans,
):
    """Test Ridge model fails with NaN values when imputation disabled."""
    model = RidgeModel(alpha=1.0, use_mean_imputation=False)
    X, y = regression_data_with_nans
    with pytest.raises(ValueError):
        model.train(X, y)


def test_ridge_model_predict_without_imputer_skips_imputation(
    regression_data,
):
    """Test Ridge model skips imputation when imputer not set."""
    model = RidgeModel(alpha=1.0, use_mean_imputation=False)
    X, y = regression_data
    model.train(X, y)
    # Verify imputer was not created
    assert model._imputer is None
    # Prediction should work fine without imputation
    preds = model.predict(X)
    assert preds.shape == (4, 1)


def test_ridge_model_serialization_with_imputation(tmp_path, regression_data_with_nans):
    """Test Ridge model serialization/deserialization with imputation."""
    model = RidgeModel(alpha=1.0, use_mean_imputation=True)
    X, y = regression_data_with_nans
    model.train(X, y)
    preds = model.predict(X)

    param_path = tmp_path / "model.json"
    serial_path = tmp_path / "model.pkl"
    model.serialize(param_path, serial_path)

    loaded_model = RidgeModel.deserialize(param_path, serial_path)
    preds_loaded = loaded_model.predict(X)
    assert_allclose(preds, preds_loaded)


def test_logistic_regression_predict_proba_with_imputation(
    classification_data_with_nans,
):
    """Test LogisticRegression predict_proba with mean imputation."""
    model = LogisticRegressionL2Model(C=1.0, use_mean_imputation=True, random_state=42)
    X, y = classification_data_with_nans
    model.train(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)
    assert not np.any(np.isnan(probs))
    assert_allclose(probs.sum(axis=1), np.ones(4))


def test_ridge_with_gridsearchcv_and_imputation():
    """Test Ridge model with GridSearchCV when imputation is enabled.

    This test verifies that the model handles the scenario where:
    1. Model has imputation enabled
    2. The underlying sklearn estimator is used in GridSearchCV
    3. Predictions work correctly after GridSearchCV completes

    This simulates how SKLearnGridSearchTrainer uses the model.
    """
    # Create training data with some variety for CV splits
    np.random.seed(42)
    X_train = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
        ]
    )
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    # Test data without NaN (since training data has no NaN)
    X_test = np.array([[25.0, 26.0, 27.0], [28.0, 29.0, 30.0]])

    # Create model with imputation enabled and build it
    model = RidgeModel(use_mean_imputation=True, alpha=1.0)
    model.build()

    # Simulate what SKLearnGridSearchTrainer does:
    # Use GridSearchCV on the underlying sklearn estimator
    # When imputation is enabled, the estimator is a Pipeline, so param names
    # must be prefixed with the pipeline step name (e.g., "model__alpha")
    sklearn_model = model.estimator
    param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(sklearn_model, param_grid=param_grid, cv=3)

    # Fit the GridSearchCV
    grid_search.fit(X_train, y_train)

    # Update the model with the best estimator (like the trainer does)
    model.estimator = grid_search.best_estimator_

    # Now predict using the model - this should work without errors
    # The imputer should be None (not used during GridSearchCV)
    # so prediction should proceed without imputation
    predictions = model.predict(X_test)

    # Verify predictions are valid
    assert predictions.shape == (2, 1)
    assert not np.any(np.isnan(predictions))
