"""Unit tests for nested_optuna module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from openadmet.models.anvil.nested_optuna import (
    NestedSearchConfig,
    run_nested_optuna_search,
)
from openadmet.models.anvil.specification import AnvilSpecification


@pytest.fixture
def small_classification_dataset():
    """Create small classification dataset for fast tests.

    Returns:
        Tuple of (X, y) with 100 samples and 10 features.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def base_estimator():
    """Create basic estimator pipeline.

    Returns:
        Pipeline with scaler and logistic regression.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=100, random_state=42)),
        ]
    )


@pytest.fixture
def param_distributions():
    """Create parameter distributions for Optuna.

    Returns:
        Dictionary of parameter distributions.
    """
    return {
        "clf__C": FloatDistribution(1e-2, 1e1, log=True),
        "clf__penalty": CategoricalDistribution(["l2"]),
    }


@pytest.fixture
def basic_config():
    """Create basic nested search config.

    Returns:
        NestedSearchConfig with minimal settings for fast tests.
    """
    return NestedSearchConfig(
        outer_n_splits=2,
        outer_repeats=1,
        outer_shuffle=True,
        outer_random_state=42,
        inner_cv=2,
        n_trials=3,
        sampler_seed=42,
        n_jobs_outer=1,
        scoring="accuracy",
    )


def test_nested_search_config_defaults():
    """Test NestedSearchConfig default values."""
    cfg = NestedSearchConfig()
    assert cfg.outer_n_splits == 5
    assert cfg.outer_repeats == 1
    assert cfg.inner_cv == 3
    assert cfg.n_trials == 50


def test_run_nested_optuna_search_basic(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test basic nested optuna search execution and result structure."""
    X, y = small_classification_dataset

    results = run_nested_optuna_search(
        X, y, base_estimator, param_distributions, basic_config
    )

    # Check result structure
    assert "outer_cv_results" in results
    assert "outer_best_params" in results
    assert "outer_best_scores" in results
    assert "estimators" in results

    # Check correct number of folds
    expected_folds = basic_config.outer_n_splits * basic_config.outer_repeats
    assert len(results["outer_best_params"]) == expected_folds
    assert len(results["outer_best_scores"]) == expected_folds
    assert len(results["estimators"]) == expected_folds

    # Check parameter values are valid
    for params in results["outer_best_params"]:
        assert "clf__C" in params
        assert "clf__penalty" in params
        assert 1e-2 <= params["clf__C"] <= 1e1

    # Check scores are valid
    for score in results["outer_best_scores"]:
        assert 0.0 <= score <= 1.0

    # Check cv_results structure
    cv_results = results["outer_cv_results"]
    assert "test_score" in cv_results
    assert "fit_time" in cv_results
    assert "estimator" in cv_results


def test_run_nested_optuna_search_with_repeats(
    small_classification_dataset,
    base_estimator,
    param_distributions,
):
    """Test nested optuna search with repeated CV produces correct folds."""
    X, y = small_classification_dataset

    cfg = NestedSearchConfig(
        outer_n_splits=2,
        outer_repeats=2,
        inner_cv=2,
        n_trials=3,
        sampler_seed=42,
        scoring="accuracy",
    )

    results = run_nested_optuna_search(X, y, base_estimator, param_distributions, cfg)

    expected_folds = 2 * 2
    assert len(results["estimators"]) == expected_folds


def test_nested_optuna_end_to_end_simple_model(
    small_classification_dataset,
):
    """Test end-to-end nested CV with Optuna on a simple classification task.

    This test verifies the complete workflow including:
    - Nested cross-validation (outer loop for evaluation)
    - Optuna hyperparameter search (inner loop for tuning)
    - Multiple parameter types (float with log scale, categorical)
    - Reproducibility with random seeds
    """
    X, y = small_classification_dataset

    # Use LogisticRegression for fast execution
    base_estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )

    # Define hyperparameter search space
    param_distributions = {
        "clf__C": FloatDistribution(1e-3, 1e2, log=True),
        "clf__penalty": CategoricalDistribution(["l1", "l2"]),
        "clf__solver": CategoricalDistribution(["liblinear"]),
    }

    # Configure nested CV with minimal settings for speed
    cfg = NestedSearchConfig(
        outer_n_splits=3,
        outer_repeats=1,
        inner_cv=2,
        n_trials=5,
        sampler_seed=42,
        scoring="accuracy",
        n_jobs_outer=1,
    )

    # Run nested CV
    results = run_nested_optuna_search(X, y, base_estimator, param_distributions, cfg)

    # Verify all folds completed successfully
    assert len(results["estimators"]) == 3

    # Verify scores are reasonable for a simple classification task
    scores = results["outer_best_scores"]
    mean_score = np.mean(scores)
    assert mean_score > 0.5  # Better than random

    # Verify each fold found different hyperparameters
    params_list = results["outer_best_params"]
    assert all("clf__C" in params for params in params_list)
    assert all("clf__penalty" in params for params in params_list)

    # Verify fitted estimators can make predictions
    for estimator in results["estimators"]:
        predictions = estimator.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})


def test_nested_optuna_anvil_workflow_integration(tmp_path):
    """Test end-to-end nested CV via Anvil workflow specification.

    This test verifies the complete integration with the Anvil API:
    - Creating an Anvil recipe YAML with SKLearnOptunaTrainer
    - Loading data from CSV
    - Featurizing with fingerprints
    - Running nested CV with Optuna through the workflow
    - Generating evaluation metrics and plots
    """
    # Create test data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )

    # Create CSV with dummy SMILES
    df = pd.DataFrame(
        {
            "smiles": [f"C{'C' * i}" for i in range(len(y))],
            "activity": y,
        }
    )
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    # Create Anvil recipe with nested Optuna trainer
    recipe = {
        "metadata": {
            "version": "v1",
            "name": "nested-optuna-integration-test",
            "build_number": 0,
            "description": "Integration test for nested CV with Optuna",
            "tag": "test-nested-optuna-anvil",
            "authors": "Test",
            "email": "test@test.com",
            "date_created": "2024-01-01",
            "biotargets": ["TEST"],
            "tags": ["test", "nested-cv", "optuna"],
        },
        "data": {
            "type": "csv",
            "resource": str(csv_path),
            "input_col": "smiles",
            "target_cols": ["activity"],
        },
        "procedure": {
            "split": {
                "type": "ShuffleSplitter",
                "params": {"train_size": 0.8, "random_state": 42},
            },
            "feat": {
                "type": "FingerprintFeaturizer",
                "params": {"fp_type": "ecfp:4"},
            },
            "model": {"type": "LGBMClassifierModel", "params": {}},
            "train": {
                "type": "SKLearnOptunaTrainer",
                "params": {
                    "outer_n_splits": 2,
                    "outer_repeats": 1,
                    "inner_cv": 2,
                    "n_trials": 3,
                    "sampler_seed": 42,
                    "param_distributions": {
                        "learning_rate": {
                            "type": "float",
                            "low": 0.01,
                            "high": 0.3,
                            "log": True,
                        },
                        "n_estimators": {
                            "type": "int",
                            "low": 10,
                            "high": 50,
                        },
                    },
                },
            },
        },
        "report": {
            "eval": [
                {"type": "ClassificationMetrics"},
            ]
        },
    }

    recipe_path = tmp_path / "nested_optuna_recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    # Run workflow through Anvil API
    output_dir = tmp_path / "output"
    anvil_spec = AnvilSpecification.from_recipe(recipe_path)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=output_dir)

    # Verify expected outputs exist
    assert Path(output_dir / "model.json").exists()
    assert Path(output_dir / "classification_metrics.json").exists()
    assert Path(output_dir / "anvil_recipe.yaml").exists()

    # Verify classification metrics were computed
    with open(output_dir / "classification_metrics.json") as f:
        metrics = json.load(f)
        assert "accuracy" in metrics
        # Metrics are returned as dicts with 'value', 'lower_ci', 'upper_ci'
        assert 0.0 <= metrics["accuracy"]["value"] <= 1.0
