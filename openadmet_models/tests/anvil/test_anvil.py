from pathlib import Path
import pytest
from openadmet_models.anvil.anvil_workflow import AnvilWorkflow, AnvilSpecification, EvalSpec
from openadmet_models.tests.datafiles import (
    basic_anvil_yaml,
    basic_anvil_yaml_featconcat,
    basic_anvil_yaml_gridsearch,
)


def all_anvil_full_recipes():
    return  [basic_anvil_yaml, basic_anvil_yaml_featconcat, basic_anvil_yaml_gridsearch]


def test_anvil_spec_create():
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    assert anvil_spec

def test_anvil_spec_create_from_recipe_roundtrip():
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    assert anvil_spec
    anvil_spec.to_recipe("tst.yaml")
    anvil_spec2 = AnvilSpecification.from_recipe("tst.yaml")
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
    assert Path(tmp_path / "tst" / "regplot.png").exists()

