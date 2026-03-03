from pathlib import Path
import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from openadmet.models.anvil import workflow as workflow_module
from openadmet.models.anvil.specification import (
    AnvilSpecification,
)
from openadmet.models.tests.unit.datafiles import (
    acetylcholinesterase_anvil_chemprop_yaml,
    anvil_yaml_featconcat,
    anvil_yaml_gridsearch,
    anvil_yaml_xgboost_cv,
    basic_anvil_yaml,
    basic_anvil_yaml_classification,
    basic_anvil_yaml_cv,
    tabpfn_anvil_classification_yaml,
)


def all_anvil_full_recipes():
    return [
        basic_anvil_yaml,
        # anvil_yaml_featconcat, # skipping as slow, redundant with integration tests
        anvil_yaml_gridsearch,
        # anvil_yaml_xgboost_cv, # skipping as slow, redundant with integration tests
    ]


def test_anvil_spec_create():
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    assert anvil_spec


def test_anvil_spec_create_from_recipe_roundtrip(tmp_path):
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    assert anvil_spec
    anvil_spec.to_recipe(tmp_path / "tst.yaml")
    anvil_spec2 = AnvilSpecification.from_recipe(tmp_path / "tst.yaml")
    # these were created from different directories, so the anvil_dir will be different
    anvil_spec.data.anvil_dir = None
    anvil_spec2.data.anvil_dir = None

    assert anvil_spec == anvil_spec2


def test_anvil_spec_create_to_workflow():
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    anvil_workflow = anvil_spec.to_workflow()
    assert anvil_workflow


def _single_split_data():
    X = pd.DataFrame({"SMILES": ["CCO"]})
    y = pd.DataFrame({"target": [1.0]})
    return X, y


def _dl_featurize_output():
    train_dataloader = [0]
    train_scaler = object()
    train_dataset = [(np.array([[1.0]]), np.array([1.0]))]
    return train_dataloader, None, train_scaler, train_dataset


def test_anvil_workflow_warns_and_skips_split_for_predefined_train_test(
    tmp_path, monkeypatch
):
    anvil_workflow = AnvilSpecification.from_recipe(basic_anvil_yaml).to_workflow()
    X_train, y_train = _single_split_data()
    X_all, y_all = _single_split_data()
    split_mock = Mock()
    anvil_workflow.data_spec = Mock(
        using_train_test=True,
        target_cols=["target"],
        read=Mock(return_value=(X_train, None, None, y_train, None, None, X_all, y_all)),
    )
    anvil_workflow.split = Mock(split=split_mock)
    anvil_workflow.feat = Mock(featurize=Mock(return_value=(np.array([[1.0]]), None)))
    anvil_workflow.model = Mock(
        _model_json_name="model.json",
        _model_save_name="model.pkl",
        serialize=Mock(),
    )
    anvil_workflow._train = Mock()
    anvil_workflow.evals = []
    anvil_workflow.transform = None
    monkeypatch.setattr(workflow_module.zarr, "save", Mock())

    with pytest.warns(
        UserWarning, match="Predefined train/test splits detected in data specification"
    ):
        anvil_workflow.run(output_dir=tmp_path / "anvil_warns")

    split_mock.assert_not_called()


def test_anvil_workflow_no_warning_when_splitter_is_used(tmp_path, monkeypatch):
    anvil_workflow = AnvilSpecification.from_recipe(basic_anvil_yaml).to_workflow()
    X_train, y_train = _single_split_data()
    X_all, y_all = _single_split_data()
    split_mock = Mock(
        return_value=(X_train, None, None, y_train, None, None, None)
    )
    anvil_workflow.data_spec = Mock(
        using_train_test=False,
        target_cols=["target"],
        read=Mock(return_value=(X_all, y_all)),
    )
    anvil_workflow.split = Mock(split=split_mock)
    anvil_workflow.feat = Mock(featurize=Mock(return_value=(np.array([[1.0]]), None)))
    anvil_workflow.model = Mock(
        _model_json_name="model.json",
        _model_save_name="model.pkl",
        serialize=Mock(),
    )
    anvil_workflow._train = Mock()
    anvil_workflow.evals = []
    anvil_workflow.transform = None
    monkeypatch.setattr(workflow_module.zarr, "save", Mock())

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        anvil_workflow.run(output_dir=tmp_path / "anvil_no_warn")

    assert not any(
        issubclass(record.category, UserWarning)
        and "Predefined train/test splits detected in data specification"
        in str(record.message)
        for record in recorded_warnings
    )
    split_mock.assert_called_once()


def test_deep_learning_workflow_warns_and_skips_split_for_predefined_train_test(
    tmp_path, monkeypatch
):
    anvil_workflow = AnvilSpecification.from_recipe(
        acetylcholinesterase_anvil_chemprop_yaml
    ).to_workflow()
    X_train, y_train = _single_split_data()
    X_all, y_all = _single_split_data()
    split_mock = Mock()
    anvil_workflow.data_spec = Mock(
        using_train_test=True,
        target_cols=["target"],
        read=Mock(return_value=(X_train, None, None, y_train, None, None, X_all, y_all)),
    )
    anvil_workflow.split = Mock(split=split_mock)
    anvil_workflow.feat = Mock(featurize=Mock(return_value=_dl_featurize_output()))
    anvil_workflow.model = Mock(
        _model_json_name="model.json",
        _model_save_name="model.pth",
        serialize=Mock(),
    )
    anvil_workflow._train = Mock()
    anvil_workflow.evals = []
    monkeypatch.setattr(workflow_module.torch, "save", Mock())

    with pytest.warns(
        UserWarning, match="Predefined train/test splits detected in data specification"
    ):
        anvil_workflow.run(output_dir=tmp_path / "anvil_deep_warns")

    split_mock.assert_not_called()


def test_deep_learning_workflow_no_warning_when_splitter_is_used(tmp_path, monkeypatch):
    anvil_workflow = AnvilSpecification.from_recipe(
        acetylcholinesterase_anvil_chemprop_yaml
    ).to_workflow()
    X_train, y_train = _single_split_data()
    X_all, y_all = _single_split_data()
    split_mock = Mock(
        return_value=(X_train, None, None, y_train, None, None, None)
    )
    anvil_workflow.data_spec = Mock(
        using_train_test=False,
        target_cols=["target"],
        read=Mock(return_value=(X_all, y_all)),
    )
    anvil_workflow.split = Mock(split=split_mock)
    anvil_workflow.feat = Mock(featurize=Mock(return_value=_dl_featurize_output()))
    anvil_workflow.model = Mock(
        _model_json_name="model.json",
        _model_save_name="model.pth",
        serialize=Mock(),
    )
    anvil_workflow._train = Mock()
    anvil_workflow.evals = []
    monkeypatch.setattr(workflow_module.torch, "save", Mock())

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        anvil_workflow.run(output_dir=tmp_path / "anvil_deep_no_warn")

    assert not any(
        issubclass(record.category, UserWarning)
        and "Predefined train/test splits detected in data specification"
        in str(record.message)
        for record in recorded_warnings
    )
    split_mock.assert_called_once()


@pytest.mark.parametrize("anvil_full_recipie", all_anvil_full_recipes())
def test_anvil_workflow_run(tmp_path, anvil_full_recipie):
    anvil_workflow = AnvilSpecification.from_recipe(anvil_full_recipie).to_workflow()
    anvil_workflow.run(output_dir=tmp_path / "tst")
    assert Path(tmp_path / "tst" / "model.json").exists()
    assert Path(tmp_path / "tst" / "regression_metrics.json").exists()
    assert any((tmp_path / "tst").glob("*regplot.png"))


def test_anvil_multiyaml(tmp_path):
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    anvil_spec.to_multi_yaml(
        metadata_yaml=tmp_path / "metadata.yaml",
        procedure_yaml=tmp_path / "procedure.yaml",
        data_yaml=tmp_path / "data.yaml",
        report_yaml=tmp_path / "eval.yaml",
    )
    anvil_spec2 = AnvilSpecification.from_multi_yaml(
        metadata_yaml=tmp_path / "metadata.yaml",
        procedure_yaml=tmp_path / "procedure.yaml",
        data_yaml=tmp_path / "data.yaml",
        report_yaml=tmp_path / "eval.yaml",
    )
    assert anvil_spec.data.anvil_dir == anvil_spec2.data.anvil_dir
    assert anvil_spec.dict() == anvil_spec2.dict()


def test_anvil_cross_val_run(tmp_path):
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml_cv)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=tmp_path / "tst")


def test_anvil_classification_run(tmp_path):
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml_classification)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=tmp_path / "tst")

    assert Path(tmp_path / "tst" / "anvil_recipe.yaml").exists()
    assert Path(tmp_path / "tst" / "model.json").exists()
    assert Path(tmp_path / "tst" / "classification_metrics.json").exists()
    assert Path(tmp_path / "tst" / "pr_curve.png").exists()
    assert Path(tmp_path / "tst" / "roc_curve.png").exists()


# skip on MacOS runner?
def test_anvil_chemprop_cpu_regression(tmp_path):
    anvil_spec = AnvilSpecification.from_recipe(
        acetylcholinesterase_anvil_chemprop_yaml
    )
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=tmp_path / "tst")
    assert Path(tmp_path / "tst" / "model.json").exists()
    assert Path(tmp_path / "tst" / "regression_metrics.json").exists()
    assert any((tmp_path / "tst").glob("*regplot.png"))


@pytest.mark.skip(reason="TabPFN requires GPU and is not supported on MacOS runners")
def test_anvil_tabpfn_classification(tmp_path):
    anvil_spec = AnvilSpecification.from_recipe(tabpfn_anvil_classification_yaml)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=tmp_path / "tst")
