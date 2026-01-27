"""Integration tests for linear models with Anvil workflows."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification, make_regression

from openadmet.models.anvil.specification import AnvilSpecification


@pytest.fixture
def regression_data(tmp_path):
    """Create regression test data.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created CSV file.
    """
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, random_state=42
    )
    df = pd.DataFrame({"smiles": [f"C{'C' * i}" for i in range(len(y))], "activity": y})
    csv_path = tmp_path / "regression_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def classification_data(tmp_path):
    """Create classification test data.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created CSV file.
    """
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=42
    )
    df = pd.DataFrame({"smiles": [f"C{'C' * i}" for i in range(len(y))], "activity": y})
    csv_path = tmp_path / "classification_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def create_anvil_recipe(tmp_path, data_path, model_type, is_classification=False):
    """Create an Anvil recipe YAML for testing.

    Args:
        tmp_path: Temporary directory path.
        data_path: Path to data CSV file.
        model_type: Type of model to use.
        is_classification: Whether this is a classification task.

    Returns:
        Path to the created recipe YAML file.
    """
    eval_type = "ClassificationMetrics" if is_classification else ("RegressionMetrics")

    recipe = {
        "metadata": {
            "version": "v1",
            "name": f"{model_type}-test",
            "build_number": 0,
            "description": f"Test {model_type}",
            "tag": f"test-{model_type.lower()}",
            "authors": "Test",
            "email": "test@test.com",
            "date_created": "2024-01-01",
            "biotargets": ["TEST"],
            "tags": ["test"],
        },
        "data": {
            "type": "csv",
            "resource": str(data_path),
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
            "model": {"type": model_type, "params": {}},
            "train": {"type": "SKLearnBasicTrainer"},
        },
        "report": {"eval": [{"type": eval_type}]},
    }

    recipe_path = tmp_path / f"{model_type}_recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    return recipe_path


def test_ridge_model_anvil_workflow(tmp_path, regression_data):
    """Test Ridge model integration with Anvil workflow."""
    recipe_path = create_anvil_recipe(
        tmp_path, regression_data, "RidgeModel", is_classification=False
    )
    output_dir = tmp_path / "ridge_output"

    anvil_spec = AnvilSpecification.from_recipe(recipe_path)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=output_dir)

    # Verify outputs exist
    assert Path(output_dir / "model.json").exists()
    assert Path(output_dir / "regression_metrics.json").exists()
    assert Path(output_dir / "anvil_recipe.yaml").exists()


def test_lasso_model_anvil_workflow(tmp_path, regression_data):
    """Test Lasso model integration with Anvil workflow."""
    recipe_path = create_anvil_recipe(
        tmp_path, regression_data, "LassoModel", is_classification=False
    )
    output_dir = tmp_path / "lasso_output"

    anvil_spec = AnvilSpecification.from_recipe(recipe_path)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=output_dir)

    # Verify outputs exist
    assert Path(output_dir / "model.json").exists()
    assert Path(output_dir / "regression_metrics.json").exists()
    assert Path(output_dir / "anvil_recipe.yaml").exists()


def test_elasticnet_model_anvil_workflow(tmp_path, regression_data):
    """Test ElasticNet model integration with Anvil workflow."""
    recipe_path = create_anvil_recipe(
        tmp_path, regression_data, "ElasticNetModel", is_classification=False
    )
    output_dir = tmp_path / "elasticnet_output"

    anvil_spec = AnvilSpecification.from_recipe(recipe_path)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=output_dir)

    # Verify outputs exist
    assert Path(output_dir / "model.json").exists()
    assert Path(output_dir / "regression_metrics.json").exists()
    assert Path(output_dir / "anvil_recipe.yaml").exists()


def test_logistic_regression_l1_anvil_workflow(tmp_path, classification_data):
    """Test LogisticRegression L1 model integration with Anvil workflow."""
    recipe_path = create_anvil_recipe(
        tmp_path,
        classification_data,
        "LogisticRegressionL1Model",
        is_classification=True,
    )
    output_dir = tmp_path / "logreg_l1_output"

    anvil_spec = AnvilSpecification.from_recipe(recipe_path)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=output_dir)

    # Verify outputs exist
    assert Path(output_dir / "model.json").exists()
    assert Path(output_dir / "classification_metrics.json").exists()
    assert Path(output_dir / "anvil_recipe.yaml").exists()


def test_logistic_regression_l2_anvil_workflow(tmp_path, classification_data):
    """Test LogisticRegression L2 model integration with Anvil workflow."""
    recipe_path = create_anvil_recipe(
        tmp_path,
        classification_data,
        "LogisticRegressionL2Model",
        is_classification=True,
    )
    output_dir = tmp_path / "logreg_l2_output"

    anvil_spec = AnvilSpecification.from_recipe(recipe_path)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=output_dir)

    # Verify outputs exist
    assert Path(output_dir / "model.json").exists()
    assert Path(output_dir / "classification_metrics.json").exists()
    assert Path(output_dir / "anvil_recipe.yaml").exists()


def test_logistic_regression_elasticnet_anvil_workflow(tmp_path, classification_data):
    """Test LogisticRegression ElasticNet model with Anvil workflow."""
    recipe_path = create_anvil_recipe(
        tmp_path,
        classification_data,
        "LogisticRegressionElasticNetModel",
        is_classification=True,
    )
    output_dir = tmp_path / "logreg_elasticnet_output"

    anvil_spec = AnvilSpecification.from_recipe(recipe_path)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=output_dir)

    # Verify outputs exist
    assert Path(output_dir / "model.json").exists()
    assert Path(output_dir / "classification_metrics.json").exists()
    assert Path(output_dir / "anvil_recipe.yaml").exists()
