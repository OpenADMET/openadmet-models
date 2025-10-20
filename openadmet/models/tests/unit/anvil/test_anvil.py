from pathlib import Path

import pytest

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
